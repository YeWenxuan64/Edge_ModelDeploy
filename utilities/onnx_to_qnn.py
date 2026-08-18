import os
import sys
import re
import json
import subprocess
from pathlib import Path
from itertools import zip_longest
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import cv2
import onnx



current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import temporary_chdir, letterbox_image, clean_files_or_dirs, read_dataset_txt_to_list
from utils import sanitize_name, parse_bitwidth, find_hybrid_subgraph_nodes
from utils import get_onnx_model_info, normalize_onnx_model, reorder_onnx_nodes_by_input, reorder_onnx_nodes_by_output
from qnn_accuracy_debugger import SnpeAccuracyDebugger
from onnx_aimet_quant import AimetOnnxQuantizer, _resolve_aimet_config_path




class QnnHybridQuantGen:
    def __init__(self, custom_hybrid:list[list[str]], weights_bitwidth:int=8, act_bitwidth:int=16, bias_bitwidth:int=8, float_bitwidth:int|None=None):
        if float_bitwidth is not None:
            if float_bitwidth not in (16, 32):
                raise ValueError('float_bitwidth must be 16 or 32')
            dtype = 'float'
            weights_bitwidth = float_bitwidth
            act_bitwidth = float_bitwidth
            bias_bitwidth = float_bitwidth
        else:
            dtype = 'int'
            if weights_bitwidth not in (4, 8, 16):
                raise ValueError('weights_bitwidth must be 4, 8 or 16')
            if act_bitwidth not in (8, 16):
                raise ValueError('act_bitwidth must be 8 or 16')
            if bias_bitwidth not in (8, 32):
                raise ValueError('bias_bitwidth must be 8 or 32')

        if not isinstance(custom_hybrid, list) or not custom_hybrid:
            raise ValueError('custom_hybrid must be a non-empty list of [input_tensor, output_tensor] pairs')

        self.hybrid_quantization = {
            "custom_hybrid": custom_hybrid,
            "dtype": dtype,
            "weights_bitwidth": weights_bitwidth,
            "act_bitwidth": act_bitwidth,
            "bias_bitwidth": bias_bitwidth,
        }
        print(f"Hybrid quantization is set: {len(custom_hybrid)} subgraph(s) dtype={dtype} "
              f"w{weights_bitwidth}a{act_bitwidth}b{bias_bitwidth}, rest quantized by global settings")

    def generate_hybrid_quantization_overrides(self, tmp_onnx_path:str) -> str|None:
        """
        根据子图的输入/输出张量生成 QAIRT 混合量化的 quantization_overrides JSON。
        每个 [输入张量, 输出张量] 对之间的所有节点按 do_hybrid_quantization 指定的
        精度标记，转换器会在子图边界自动插入 Convert 节点。

        Returns:
            str | None: overrides JSON 文件路径；失败返回 None。
        """
        try:
            model = onnx.load_model(tmp_onnx_path)
        except Exception as e:
            print(f"Error loading ONNX model for hybrid quantization: {e}")
            return None

        nodes = list(model.graph.node)

        # 识别子图节点：每个 [输入张量, 输出张量] 对 -> 输入下游 ∩ 输出上游的节点并集
        # （复用共享的 utils.find_hybrid_subgraph_nodes，与 AIMET 路径的搜索逻辑一致）
        try:
            middle = find_hybrid_subgraph_nodes(model, self.hybrid_quantization["custom_hybrid"])
        except ValueError as e:
            print(f"Error: {e}")
            return None

        if not middle:
            print("Error: no nodes selected for hybrid quantization")
            return None

        hq = self.hybrid_quantization
        dtype = hq["dtype"]
        act_bw = hq["act_bitwidth"]
        weight_bw = hq["weights_bitwidth"]
        bias_bw = hq["bias_bitwidth"]

        initializer_names = {init.name for init in model.graph.initializer}
        activation_encodings: dict[str, list[dict]] = {}
        param_encodings: dict[str, list[dict]] = {}

        # 识别 bias: 作为 Conv/ConvTranspose/Gemm 第3个输入(index 2)的 initializer
        bias_names = set()
        for n in nodes:
            if n.op_type in ('Conv', 'ConvTranspose', 'Gemm') and len(n.input) > 2:
                bias_names.add(n.input[2])

        for idx in middle:
            n = nodes[idx]
            for out in n.output:
                if out and out not in activation_encodings:
                    activation_encodings[out] = [{"bitwidth": act_bw, "dtype": dtype}]
            for inp in n.input:
                if inp in initializer_names and inp not in param_encodings:
                    param_bw = bias_bw if inp in bias_names else weight_bw
                    param_encodings[inp] = [{"bitwidth": param_bw, "dtype": dtype}]

        overrides = {
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
            "version": "0.5.0",
        }

        overrides_path = Path(tmp_onnx_path).parent / "quantization_overrides.json"
        with open(str(overrides_path), 'w') as f:
            json.dump(overrides, f, indent=4)

        print(f"Hybrid quantization overrides generated: {len(middle)} nodes, "
              f"{len(activation_encodings)} activation tensors, {len(param_encodings)} param tensors -> {overrides_path}")
        return str(overrides_path)


class QnnAimetConnector:
    """AIMET 2.x 量化路径连接器（OnnxToQNN.set_use_aimet 创建）。

    封装 AIMET 量化路径：AIMET PTQ -> QDQ ONNX + encodings -> qairt-converter
    转量化 DLC -> context binary（可选精度分析）。OnnxToQNN.convert() 检测到
    self.aimet_connector 后调用其 convert()。
    """

    def __init__(self, converter:'OnnxToQNN', quant_method:str, bitwidth:str, param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric',
                 encoding_version:str='2.0.0', config_file:str|None=None):
        """创建连接器并保存 AIMET 量化配置。

        Args:
            converter: 父级 OnnxToQNN；回调 convert_onnx_model / write_config_file /
                generate_context_binary_model 等，并读取校准数据 / 精度分析器状态。
            quant_method: AIMET 方案（'min_max'/'tf_enhanced'/'percentile' 及别名）。
            bitwidth: 全局位宽 'w<W>a<A>'，如 'w8a8'。
            param_quant_schema: 权重对称性（'asymmetric'/'symmetric'/'unsignedsymmetric'）。
                默认 'symmetric'。
            act_quant_schema: 激活对称性（'asymmetric'/'symmetric'/'unsignedsymmetric'）。
                默认 'asymmetric'。
            encoding_version: encodings 版本（'0.6.1'/'1.0.0'/'2.0.0'）。默认 '2.0.0'。
            config_file: AIMET quantsim_config 路径或别名；传 htp 版本（如 'htp_v68'）
                会解析为对应内置配置绝对路径；None 用 'default'。
        """
        if param_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('param_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')
        
        if act_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('act_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')

        self.converter = converter
        self.quant_method = quant_method
        self.bitwidth = bitwidth
        self.param_quant_schema = param_quant_schema
        self.act_quant_schema = act_quant_schema
        self.encoding_version = encoding_version

        # 传入 htp 版本（如 'htp_v68'/'htp_v73'）时，在此加载对应版本的内置
        # quantsim_config，得到其绝对路径。对称性等后续由 AimetOnnxQuantizer.
        # _build_sim 基于该内置配置改写（defaults 级）后应用，绝不改动算子级配置。
        self.config_file = (_resolve_aimet_config_path(config_file)
                            if config_file is not None else None)
        if self.config_file is not None:
            print(f"[AIMET] load quantsim_config: {self.config_file}")

        print(f"Enabled AIMET 2.x quantization path (scheme={self.quant_method}, {bitwidth}")

    def current_hybrid_config(self) -> tuple:
        """实时读取 converter.hybrid_quantizer 的混合量化配置（与调用顺序无关）。

        Returns:
            (hybrid_subgraphs, hybrid_weights_bitwidth, hybrid_act_bitwidth,
             hybrid_float_bitwidth)；未设置时返回 (None, 8, 16, None)。
        """
        hybrid_quantizer = self.converter.hybrid_quantizer
        if hybrid_quantizer is None:
            return None, "w8a16", None
        
        hq = hybrid_quantizer.hybrid_quantization
        hybrid_subgraphs = hq["custom_hybrid"]
        hybrid_bitwidth = f"w{hq['weights_bitwidth']}a{hq['act_bitwidth']}"

        if hq["dtype"] == "float": 
            return (hybrid_subgraphs, hybrid_bitwidth, hq["weights_bitwidth"]) # 浮点保留模式：w/a/b 同为 float 位宽
        
        return hybrid_subgraphs, hybrid_bitwidth, None

    def convert(self, onnx_model_info:dict, mean_rgb:list, std_rgb:list, set_input_order:str):
        """执行 AIMET 量化路径：AIMET PTQ -> QDQ ONNX + encodings -> 量化 DLC
        -> context binary，可选精度分析。由 OnnxToQNN.convert() 调用。

        Args:
            onnx_model_info: ONNX 模型信息（get_onnx_model_info 产出，含 inputs 等）。
            mean_rgb / std_rgb: 每个输入的 RGB 归一化参数（精度分析用）。
            set_input_order: 输入布局，'nhwc'/'nchw'（校准 .raw 的布局解释）。
        """
        converter = self.converter
        tmp_onnx_path = converter.tmp_onnx_path
        tmp_dir = converter.tmp_dir

        # 混合量化实时读取 converter.hybrid_quantizer（兼容 do_hybrid_quantization
        # 在 set_use_aimet 之后调用的顺序），避免 set_use_aimet 快照时 hybrid 尚未设置
        quantizer = AimetOnnxQuantizer(str(tmp_onnx_path), config_file=self.config_file)

        quantizer.set_quantization_method(self.quant_method, self.bitwidth,
                                          param_quant_schema=self.param_quant_schema, act_quant_schema=self.act_quant_schema)


        hybrid_subgraphs, hybrid_bitwidth, hybrid_float_bitwidth = self.current_hybrid_config()
        if hybrid_subgraphs:
            quantizer.do_hybrid_quantization(hybrid_subgraphs, hybrid_bitwidth, hybrid_float_bitwidth)

        # AIMET 量化必须有真实校准数据：随机 dummy 无法反映真实激活分布，直接报错
        if converter.custom_alibration_data_path is None and converter.dataset_path is None:
            raise ValueError(
                "AIMET quantization requires calibration data: provide dataset_path or "
                "call use_custom_alibration_data(path) before convert()."
            )

        # 生成 AIMET 校准输入（与 QAIRT 校准数据使用同一套预处理）
        calibration_inputs = self.generate_calibration_inputs(onnx_model_info, set_input_order)
        quantizer.quantize(calibration_data=calibration_inputs)

        # 导出 QDQ ONNX + encodings
        qdq_path, enc_path = quantizer.export(
            str(tmp_dir),
            f"{tmp_onnx_path.stem}_aimet_qdq",
            encoding_version=self.encoding_version,
        )
        converter.file_or_dir_to_clean.append(qdq_path)
        converter.file_or_dir_to_clean.append(enc_path)

        # AIMET 导出的 QDQ 输入消费链排列顺序可能与 graph.input 不一致（多输入模型常见），
        # 会导致 qairt-converter 推导出的 DLC 输入顺序错位（运行时按 graph.input 顺序
        # 喂数据时 NPU 输入错乱）。按 graph.input 顺序重排 QDQ 输入链后再转换。
        self.reorder_qdq_input_chains_by_graph_order(qdq_path)

        # 转换 QDQ ONNX -> 量化 DLC（AIMET 编码直接进入 DLC，无需 qairt-quantizer）
        # is_quantized=True 时 convert_onnx_model 自动给 DLC 命名加 _quantized 后缀
        dlc_model_path = converter.convert_onnx_model(onnx_model_info, set_input_order,
                                                      input_network_path=qdq_path, is_quantized=True)
        if dlc_model_path is None:
            exit(1)

        config_path = converter.write_config_file(dlc_model_path)
        converter.generate_context_binary_model(dlc_model_path, config_path)

        if converter.accuracy_analyzer is not None and dlc_model_path is not None:
            # 精度分析 golden：纯浮点 DLC
            golden_dlc_path = converter.convert_onnx_model(onnx_model_info, set_input_order,
                                                           input_network_path=str(tmp_onnx_path),
                                                           output_dlc_name=f"{tmp_onnx_path.stem}_golden")

            converter.accuracy_analyzer.set_model_inof(onnx_model_info, golden_dlc_path, dlc_model_path)
            return_code = converter.accuracy_analyzer.accuracy_analysis(mean_rgb, std_rgb, set_input_order)

            if return_code == 0:
                print("Accuracy analysis completed successfully.")
            else:
                print("Accuracy analysis failed.")

    def generate_calibration_inputs(self, onnx_model_info:dict, set_input_order:str):
        """生成 AIMET 校准输入（始终为 NCHW）。

        优先级：1) custom_alibration_data_path（.raw，布局由 set_input_order 解释，
        'nhwc' 先按 NHWC reshape 再转 NCHW）；2) dataset_path（图片 letterbox NCHW）；
        3) 都没有 -> 报错。

        Args:
            onnx_model_info: ONNX 模型信息（含 inputs，用于输入名 / shape）。
            set_input_order: 输入布局，'nhwc'/'nchw'（决定 .raw 的 reshape 与转置）。

        Yields:
            每个样本为 {输入名: np.ndarray}（NCHW）。
        """
        converter = self.converter
        input_infos = onnx_model_info["inputs"]
        needs_nhwc_to_nchw = (set_input_order == 'nhwc')

        # 1) 自定义 .raw 校准数据
        if converter.custom_alibration_data_path is not None:
            raw_dir = Path(converter.custom_alibration_data_path).parent
            with open(str(converter.custom_alibration_data_path)) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for line in lines:
                raws = [p for p in line.split(' ') if p]
                if len(raws) != len(input_infos):
                    print(f"Warning: line has {len(raws)} files but model has {len(input_infos)} inputs, skipped")
                    continue
                sample = {}
                for idx, info in enumerate(input_infos):
                    model_shape = [d if d != 'dynamic' else 1 for d in info['shape']]
                    p = Path(raws[idx])
                    if not p.is_absolute():
                        p = raw_dir / p
                    arr = np.fromfile(str(p), dtype=np.float32)
                    # 自定义 .raw 布局由 set_input_order 决定：
                    #   'nhwc' -> 用户数据为 NHWC [N,H,W,C]，先按 NHWC reshape 再转 NCHW
                    #   'nchw' -> 用户数据即 NCHW，直接 reshape 为模型输入 shape
                    if needs_nhwc_to_nchw and len(model_shape) == 4:
                        target_shape = [model_shape[0], model_shape[2], model_shape[3], model_shape[1]]
                    else:
                        target_shape = model_shape
                    try:
                        arr = arr.reshape(target_shape)
                    except ValueError:
                        print(f"Warning: {p} cannot be reshaped to {target_shape}, skipped")
                        sample = {}
                        break
                    if needs_nhwc_to_nchw and len(model_shape) == 4:
                        arr = np.transpose(arr, (0, 3, 1, 2))
                    sample[info['name']] = np.ascontiguousarray(arr)
                if sample:
                    yield sample
            return

        # 2) 图片数据集
        if converter.dataset_path is not None:
            dataset_dir = converter.dataset_path.parent
            with open(str(converter.dataset_path)) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for line in lines:
                img_paths = [p for p in line.split(' ') if p]
                if len(img_paths) != len(input_infos):
                    print(f"Warning: line has {len(img_paths)} images but model has {len(input_infos)} inputs, skipped")
                    continue
                sample = {}
                for idx, info in enumerate(input_infos):
                    shape = info['shape']
                    if len(shape) != 4 or shape[0] != 1:
                        raise ValueError(f"Unsupported input shape {shape} for input {info['name']}")
                    height, width = shape[2], shape[3]
                    p = Path(img_paths[idx])
                    if not p.is_absolute():
                        p = dataset_dir / p
                    img = cv2.imread(str(p))
                    if img is None:
                        print(f"Warning: could not read {p}")
                        sample = {}
                        break
                    arr = letterbox_image(img, (width, height),
                                          output_format='nchw',
                                          output_dtype='float32')
                    sample[info['name']] = np.ascontiguousarray(arr)
                if sample:
                    yield sample
            return

        # 3) 未提供任何校准数据：直接报错（随机 dummy 无法反映真实分布，量化编码无意义）
        raise ValueError(
            "No calibration data provided for AIMET quantization. "
            "Provide dataset_path or call use_custom_alibration_data(path) "
            "before convert()."
        )

    @staticmethod
    def reorder_qdq_input_chains_by_graph_order(qdq_model_path:str, aggressive:bool=True) -> str:
        """按 graph.input 顺序重排 QDQ 输入消费链，保证 qairt-converter 推导的 DLC
        输入顺序与运行时一致（多输入模型，如 lightFC）。仅调整节点顺序、不改数值。

        Args:
            qdq_model_path: QDQ ONNX 路径（原地重排保存）。
            aggressive: 是否激进重排（回传 weight_dq 分支优先级）。默认 True。

        Returns:
            str: 重排后的 QDQ ONNX 路径。
        """
        model = onnx.load_model(qdq_model_path)
        graph = model.graph
        input_names = [i.name for i in graph.input]
        init_names = {init.name for init in graph.initializer}
        real_inputs = [n for n in input_names if n not in init_names]
        if len(real_inputs) < 2:
            return qdq_model_path

        model = reorder_onnx_nodes_by_input(model, 5, aggressive=aggressive)
        model = reorder_onnx_nodes_by_output(model, 10, aggressive=aggressive)

        onnx.checker.check_model(model, full_check=True)
        onnx.save_model(model, qdq_model_path)
        print(f"[AIMET] Reordered QDQ input chains to match graph input order: {real_inputs}"
              f" (aggressive={aggressive})")
        return qdq_model_path



class OnnxToQNN:
    def __init__(self, model_path:str, qnn_model_path:str, dataset_path:str|None=None, target_platform:str='qcs6490'):
        """
        Initialize the ONNX to QNN converter.

        Args:
            model_path (str): Path to the input ONNX model file that needs to be converted.

            qnn_model_path (str): Path where the converted QNN model file will be saved.

            dataset_path (str | None): Path to a text file containing paths to dataset images for quantization. 
                - The text file should contain one image path per line for single-input models, 
                or multiple image paths separated by spaces for multi-input models.
                - Default is None. no quantization will be performed.

            target_platform (str): Target platform for the QNN model.
                - Available options: 'qcs6490', 'qcs8550', 'qcs9075'.
                - Default: 'qcs6490'.
        """

        self.model_path = Path(model_path).resolve()
        self.qnn_model_path = Path(qnn_model_path).resolve()

        self.dataset_path = dataset_path
        if dataset_path:
            self.dataset_path = Path(dataset_path).resolve()

        self.target_platform = target_platform
        self.architecture_dict = {
            "qcs6490": {"dsp_arch": "v68", "soc_id": 35},
            "qcs8550": {"dsp_arch": "v73", "soc_id": 66},
            "qcs9075": {"dsp_arch": "v73", "soc_id": 77},
            "SC8280X": {"dsp_arch": "v68", "soc_id": 37},
        }

        if target_platform not in self.architecture_dict.keys():
            raise ValueError(f"Invalid target platform: {target_platform}. Available options: {self.architecture_dict.keys()}")


        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径

        qairt_path = Path(current_dir).resolve() / 'qairt'
        version_dir = next(qairt_path.iterdir())# 获取qairt目录下的第一个子目录
        self.qnn_sdk_dir = version_dir

        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径

        sanitize_model_name = sanitize_name(self.model_path.stem)
        tmp_onnx_path = self.tmp_dir / sanitize_model_name
        self.tmp_onnx_path = tmp_onnx_path.with_suffix('.onnx')

        self.file_or_dir_to_clean = []
        self.accuracy_analyzer = None
        self.hybrid_quantizer = None
        self.aimet_connector = None

        #self.set_quantization_method()
        self.param_quant_method, self.act_quant_method = 'min-max', 'min-max'
        self.weights_bitwidth, self.act_bitwidth = 8, 8
        self.bias_bitwidth = 8
        self.param_quant_schema, self.act_quant_schema = 'asymmetric', 'asymmetric'
        self.use_cle_algorithm = False

        #self.use_custom_alibration_data()
        self.custom_alibration_data_path = None

    def set_quantization_method(self, param_quant_method:str='min-max', act_quant_method:str='min-max', bitwidth:str='w8a8', bias_bitwidth:int=8,
                                param_quant_schema:str='asymmetric', act_quant_schema:str='asymmetric', use_cle_algorithm:bool=False):
        """
        Configure quantization parameters for the model.
        
        Args:
            param_quant_method (str): Quantization method for model parameters (weights).
                - Available options: 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'.
                - Default: 'min-max'.

            act_quant_method (str): Quantization method for activations.
                - Available options: 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'.
                - Default: 'min-max'.

            bitwidth (str): Quantization bitwidth configuration in format 'w<W>a<A>', 
                where W is weight bitwidth and A is activation bitwidth.
                - Available options: 'w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16'.
                - Default: 'w8a8'.

            bias_bitwidth (int): Bitwidth for bias quantization.
                - Available options: 8, 32.
                - Default: 8.

            param_quant_schema (str): Parameter(weight) quantization schema
                - Available options: 'asymmetric', 'symmetric', 'unsignedsymmetric'.
                - Default: 'asymmetric'.

            act_quant_schema (str): Activation quantization schema
                - Available options: 'asymmetric', 'symmetric', 'unsignedsymmetric'.
                - Default: 'asymmetric'.
            use_cle_algorithm (bool): Whether to use the Cross Layer Equalization algorithm for quantization.
        """

        if param_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('param_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if act_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('act_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')
        
        if bias_bitwidth not in [8, 32]:
            raise ValueError('bias_bitwidth must be 8 or 32')

        if param_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('param_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')
        
        if act_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('act_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')
        

        self.param_quant_method = param_quant_method
        self.act_quant_method = act_quant_method

        self.weights_bitwidth, self.act_bitwidth = parse_bitwidth(bitwidth)

        self.bias_bitwidth = bias_bitwidth
        self.param_quant_schema = param_quant_schema

        self.act_quant_schema = act_quant_schema
        self.use_cle_algorithm = use_cle_algorithm
        
        print(f"[QnnxToQNN] Quantization method set to: quant_method: param={self.param_quant_method}, act={self.act_quant_method}; bitwidth={self.weights_bitwidth}w{self.act_bitwidth}a"
              f", schema: act={self.act_quant_schema}, param={self.param_quant_schema}; use_cle_algorithm={self.use_cle_algorithm}")

    def use_custom_alibration_data(self, custom_alibration_data_path:str|None=None):
        """
        Args:
            custom_alibration_data_path (str | None): Path to a text file containing the custom calibration dataset.
                - Each line in the text file should represent a path to image data.
                - If the model has multiple inputs, the paths should be separated by spaces.
                
                - The calibration data must be preprocessed to match the model's input dimensions, format, and data type.
                - The data must be in .raw binary format generated by np.ndarray.tofile().
                
                - Example: If the model input is float32 data with shape [1, 3, 224, 224], 
                the images must be preprocessed to match this shape and data type before being converted to .raw format.
                ```
                resized_image = cv2.resize(image, (224, 224))
                tranposed_image = np.transpose(resized_image, (2, 0, 1))
                batched_image = np.expand_dims(tranposed_image, axis=0)
                np.float32(batched_image).tofile('image.raw')
                ```
        """

        if custom_alibration_data_path is None:
            self.custom_alibration_data_path = None
        else:
            self.custom_alibration_data_path = Path(custom_alibration_data_path).resolve()

        print(f"[QnnxToQNN] Custom calibration dataset path set to: {self.custom_alibration_data_path}")

    def do_hybrid_quantization(self, custom_hybrid:list[list[str, str]], bitwidth:str="w8a16", bias_bitwidth:int=8, float_bitwidth:int|None=None):
        """
        设置混合量化(与 onnx_to_rknn.py 的 do_hybrid_quantization 一致)：
        通过子图的输入张量与输出张量指定区域，自动识别两者之间的所有节点，
        对这些节点使用指定精度，子图之外的节点仍按 set_quantization_method
        的全局设置(默认 w8a8)量化为 INT8。

        两种模式(二选一)：
        1. 整数混合量化(默认)：weights_bitwidth / act_bitwidth / bias_bitwidth
           指定区域内权重 / 激活 / 偏置的整数位宽。
           例如全局 w8a8、区域 w8a16：weights_bitwidth=8, act_bitwidth=16。
        2. 浮点保留：float_bitwidth 指定区域保持浮点精度，16=FP16，32=FP32。

        Args:
            custom_hybrid (list[list[str]]): 每个内层列表为 [输入张量名, 输出张量名]，
                表示一个混合量化子图：输入张量与输出张量之间的所有节点被选中。
                可传入多个子图，例如 [[in1, out1], [in2, out2]]。
                张量名也可以是节点名(自动取该节点的输出张量作为边界)。

            bitwidth (str): Quantization bitwidth configuration in format 'w<W>a<A>', 
                where W is weight bitwidth and A is activation bitwidth.
                - Available options: 'w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16'.
                - Default: 'w8a16'.

            bias_bitwidth (int): 整数模式下区域内的偏置位宽，可选 8/32。默认 8。
            float_bitwidth (int | None): 若设置(16/32)，区域保持浮点(FP16/FP32)，
                此时忽略三个整数位宽参数。默认 None 表示使用整数混合量化。
        """

        weights_bitwidth, act_bitwidth = parse_bitwidth(bitwidth)
        self.hybrid_quantizer = QnnHybridQuantGen(custom_hybrid, weights_bitwidth, act_bitwidth, bias_bitwidth, float_bitwidth)

        print(f"[QnnxToQNN] Hybrid quantization set to: {custom_hybrid}, bitwidth={bitwidth}, bias_bitwidth={bias_bitwidth}, float_bitwidth={float_bitwidth}")

    def set_use_aimet(self, quant_method:str='tf_enhanced', bitwidth:str="w8a8", param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric',
                      encoding_version:str='2.0.0'):
        """启用 AIMET 2.x 量化路径（替代 QAIRT 自带的 qairt-quantizer 校准）。

        启用后 convert() 流程变为：
            ONNX（已烘焙归一化）-> AIMET PTQ 量化 -> QDQ ONNX + encodings
            -> qairt-converter 直接转量化 DLC -> context binary

        只创建并保存 QnnAimetConnector 连接器；AIMET 量化、QDQ 输入链重排、
        DLC 转换、精度分析等具体流程封装在连接器内（参数说明见连接器）。

        Args:
            quant_method: AIMET 方案 'min_max'/'tf_enhanced'/'percentile'
                （含别名 'min-max'/'minmax'/'tf'/'tf-enhanced'）。默认 'tf_enhanced'。
            bitwidth: AIMET 全局位宽 'w<W>a<A>'，如 'w8a8'/'w8a16'。默认 'w8a8'。
            param_quant_schema: 权重对称性 'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'symmetric'。
            act_quant_schema: 激活对称性 'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'asymmetric'。
            encoding_version: AIMET encodings 版本 '0.6.1'/'1.0.0'/'2.0.0'。默认 '2.0.0'。

        说明：
            - 混合精度不在此传入：先调用 do_hybrid_quantization() 指定子图与精度，
              本方法在 convert 时自动读取。
            - quantsim config 无需手动指定：按 self.target_platform 的 dsp_arch
              自动选用对应 HTP config（'htp_v68'/'htp_v73'...），贴合目标硬件。
            - set_quantization_method 中 param/act 校准方法会映射为 AIMET 方案，
              但优先级低于这里显式传入的 quant_method。
        """
        # 创建连接器：归一化量化方案别名（'min-max' -> 'min_max'、'tf' -> 'tf_enhanced' 等）
        # 在 QnnAimetConnector.__init__ 内尽早校验并统一存储；混合精度由连接器在
        # convert 时通过 current_hybrid_config() 实时读取（与 do_hybrid_quantization
        # 的调用顺序无关，即使在其之前调用也能生效）。
        # 根据目标平台 DSP 架构自动选用 AIMET HTP quantsim config（'htp_v68'/'htp_v73'...），
        # 针对 HTP 后端做算子级量化约束优化（比默认 default_config 更贴合目标硬件）。
        dsp_arch = self.architecture_dict[self.target_platform]["dsp_arch"]
        config_file = f"htp_{dsp_arch}"

        self.aimet_connector = QnnAimetConnector(
            self,
            quant_method=quant_method,
            bitwidth=bitwidth,
            param_quant_schema=param_quant_schema,
            act_quant_schema=act_quant_schema,
            encoding_version=encoding_version,
            config_file=config_file,
        )

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]|None=None):
        """
        Args:
            accuracy_analysis_picture_list (list[str], optional): A list of image paths required for model accuracy analysis. 
                - Each element in the list should be a path to an image. 
                - For models with a single input, provide a single image path. 
                - For models with multiple inputs, provide multiple image paths. Example: ['/home/xxx/1.jpg', '/home/xxx/2.jpg']
                - Defaults to None.
        """

        self.accuracy_analyzer = SnpeAccuracyDebugger(self.tmp_dir, self.tmp_onnx_path, accuracy_analysis_picture_list, self.run_subprocess)

        print(f"[QnnxToQNN] Accuracy analysis data list set to: {accuracy_analysis_picture_list}")


    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]], set_input_order:str='nhwc'):
        """
        Args:
            mean_rgb (list[list[int | float,]], optional): Mean values for RGB channels normalization.
                - Each inner list contains 3 values (R, G, B) representing the mean for each channel in one input.
                - If multiple inputs are provided, For example, [[123, 116, 103], [123, 116, 103]]
                - Defaults to [[0, 0, 0]] (no mean normalization).
                
            std_rgb (list[list[int | float,]], optional): Standard deviation values for RGB channels normalization.
                - Each inner list should contain 3 values (R, G, B) representing the standard deviation for each channel in one input.
                - Similar to mean_rgb, can provide multiple lists for multiple inputs.
                - Defaults to [[1, 1, 1]] (no standard deviation normalization).

            set_input_order (str, optional): Input order for the converted model. 'nhwc' or 'nchw'.
                - Defaults to 'nhwc'.
        """

        # 1.
        self.run_env_script()

        # 2.
        self.modify_onnx_model(mean_rgb, std_rgb)

        # 3.
        onnx_model_info = get_onnx_model_info(self.tmp_onnx_path)
        if onnx_model_info is None:
            exit(1)

        # 4.1 AIMET 2.x 量化路径：AIMET 量化出 QDQ ONNX -> qairt-converter 转量化 DLC
        if self.aimet_connector is not None:
            self.aimet_connector.convert(onnx_model_info, mean_rgb, std_rgb, set_input_order)
            return

        # 4.2
        if self.hybrid_quantizer is not None:
            quantization_overrides_path = self.hybrid_quantizer.generate_hybrid_quantization_overrides(self.tmp_onnx_path)
        else:
            quantization_overrides_path = None

        # 5.
        dlc_model_path = self.convert_onnx_model(onnx_model_info, set_input_order, quantization_overrides_path)
        if dlc_model_path is None:
            exit(1)

        # 6.
        if self.dataset_path is not None and self.custom_alibration_data_path is None:
            calibration_data_index_path = self.generate_calibration_data(onnx_model_info, set_input_order)
        else:
            calibration_data_index_path = self.custom_alibration_data_path

        # 7.
        if calibration_data_index_path is not None:
            quantized_dlc_model_path = self.quantize_model(dlc_model_path, calibration_data_index_path)
        else:
            quantized_dlc_model_path = dlc_model_path

        if quantized_dlc_model_path is None:
            exit(1)

        # 8.
        config_path = self.write_config_file(dlc_model_path)

        # 9.
        self.generate_context_binary_model(quantized_dlc_model_path, config_path)

        # 10. accuracy_analyze
        if self.accuracy_analyzer is not None and quantized_dlc_model_path is not None:
            if self.hybrid_quantizer is None:
                golden_dlc_path = dlc_model_path
            else:
                # 混合量化时，精度分析的 golden 参考必须使用纯浮点 DLC：
                golden_dlc_path = self.convert_onnx_model(onnx_model_info, set_input_order, None, output_dlc_name=f"{self.tmp_onnx_path.stem}_golden")

            self.accuracy_analyzer.set_model_inof(onnx_model_info, golden_dlc_path, quantized_dlc_model_path)
            return_code = self.accuracy_analyzer.accuracy_analysis(mean_rgb, std_rgb, set_input_order)

            if return_code == 0:
                print("Accuracy analysis completed successfully.")
            else:
                print("Accuracy analysis failed.")

    def clean(self):
        """清理本次转换产生的临时文件/目录: file_or_dir_to_clean 中登记的项"""
        clean_files_or_dirs(self.file_or_dir_to_clean)

        if self.accuracy_analyzer:
            self.accuracy_analyzer.clean()


    @staticmethod
    def run_subprocess(command:str) -> int:
        executable = '/bin/bash'
        print(f"Running command: {command}")

        # 使用实时输出的方式执行命令
        process = subprocess.Popen(command, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                                universal_newlines=True, executable=executable, env=os.environ)
        
        # 实时打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取返回码
        return_code = process.poll()

        return return_code

    def run_env_script(self):
        # Linux/Unix系统使用bash脚本
        envsetup_script = self.qnn_sdk_dir / 'bin/envsetup.sh'
        command = f"source '{envsetup_script}' && env"
        executable = '/bin/bash'
        encoding = 'utf-8'
        print("Setting up QAIRT Linux environment...")
            
        # 执行脚本
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable=executable)
        
        stdout, stderr = proc.communicate() # 获取输出
        
        if proc.returncode != 0:
            print(f"Error executing script: {stderr.decode(encoding)}")
            return False
        
        # 解析环境变量
        for line in stdout.decode().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
        
    def modify_onnx_model(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        self.tmp_dir.mkdir(exist_ok=True) # 确保tmp目录存在

        if not self.model_path.exists():
            print(f"Error: ONNX file not found at {self.model_path}")
            return None
        
        model = onnx.load_model(str(self.model_path))

        # 归一化
        model = normalize_onnx_model(model, mean_rgb, std_rgb)

        model = reorder_onnx_nodes_by_input(model, 5)
        model = reorder_onnx_nodes_by_output(model, 10)

        onnx.checker.check_model(model, full_check=True)
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)

        # 复制ONNX文件到tmp目录
        onnx.save_model(model, str(self.tmp_onnx_path))
        self.file_or_dir_to_clean.append(self.tmp_onnx_path)
        print(f"Copied ONNX file to {self.tmp_onnx_path}")

    def convert_onnx_model(self, onnx_model_info:dict, set_input_order:str, quantization_overrides_path:str|None=None, output_dlc_name:str|None=None, input_network_path:str|None=None, is_quantized:bool=False) -> str|None:
        """
        Args:
            onnx_model_info (dict): 模型输入输出信息。
            set_input_order (str): 'nhwc' 或 'nchw'。
            quantization_overrides_path (str | None): QAIRT quantization_overrides JSON。
            output_dlc_name (str | None): 输出 DLC 文件名（不含后缀）。
            input_network_path (str | None): 输入 ONNX 路径；None 使用 self.tmp_onnx_path。
            is_quantized (bool): 输入是否为已量化(QDQ) ONNX。True 时不再追加
                --float_bitwidth（QDQ 模型自带编码），且未指定 output_dlc_name 时
                输出名自动带 _quantized 后缀（与 QAIRT 标准路径 quantize_model 一致）。
        """
        layout_params = [] # 构建输入布局参数
        for input_info in onnx_model_info.get("inputs"): 
            input_name = input_info["name"]
            
            if set_input_order == 'nhwc': # 为每个输入添加源布局和目标布局参数
                layout_params.extend([f'--source_model_input_layout "{input_name}" NCHW', f'--desired_input_layout "{input_name}" NHWC'])
                
            layout_params.extend([f'--desired_input_color_encoding "{input_name}" rgb rgb'])
        
        layout_args = " ".join(layout_params) # 将布局参数拼接成字符串


        extra_args = "--target_backend HTP --onnx_skip_simplification " # --onnx_summary' # --preserve_onnx_output_order

        if not is_quantized and not self.dataset_path and not self.custom_alibration_data_path:
            extra_args += " --float_bitwidth 16"

        if quantization_overrides_path:
            self.file_or_dir_to_clean.append(quantization_overrides_path)

            extra_args += f" --quantization_overrides {quantization_overrides_path}"

            if self.hybrid_quantizer is not None:
                hybrid_quantization_dict = self.hybrid_quantizer.hybrid_quantization
                if hybrid_quantization_dict["dtype"] == "float":
                    extra_args += f" --float_bitwidth {hybrid_quantization_dict['weights_bitwidth']}"


        if output_dlc_name is None and is_quantized:
            # 量化(QDQ) DLC 自动带 _quantized 后缀（与 QAIRT 标准路径 quantize_model 输出一致）
            output_dlc_name = f"{self.tmp_onnx_path.stem}_quantized"

        if output_dlc_name is not None:
            dlc_path = self.tmp_onnx_path.parent / f"{output_dlc_name}.dlc"
        else:
            dlc_path = self.tmp_onnx_path.with_suffix('.dlc')

        if input_network_path is None:
            input_network_path = str(self.tmp_onnx_path)

        command = f"qairt-converter --input_network {input_network_path} {layout_args} {extra_args} -o {dlc_path}"

        return_code = self.run_subprocess(command)
        
        if return_code == 0:
            print("Convert onnx to qnn-dlc successful!")

            self.file_or_dir_to_clean.append(dlc_path)
            return dlc_path
        
        else:
            return None
        
    def generate_calibration_data(self, onnx_model_info:dict, set_input_order:str) -> str|None:
        """
        生成校准数据
        
        Args:
            onnx_model_info (dict): 模型信息，包含输入尺寸
        
        Returns:
            list[str] | None: 校准数据文件路径列表，每个输入对应一个文件
        """

        dataset_path_list = read_dataset_txt_to_list(self.dataset_path)
        
        try:
            calibration_files = [] # 为每个输入创建目录和文件列表
            for idx, input_info in enumerate(onnx_model_info["inputs"]):
                # 创建输出目录
                output_dir = self.tmp_dir / f"calibration_data_for_input{idx + 1}"
                output_dir.mkdir(parents=True, exist_ok=True)
                self.file_or_dir_to_clean.append(output_dir)

                # 获取当前输入的尺寸
                input_shape = input_info["shape"]
                if len(input_shape) != 4 or input_shape[0] != 1:
                    print(f"Error: Unsupported input shape for input {idx + 1}")
                    continue

                height, width = input_shape[2], input_shape[3]

                def to_file_thread(img: np.ndarray, output_path: str):
                    img.tofile(output_path)

                max_workers = min(16, len(dataset_path_list))
                Threadpool_to_file = ThreadPoolExecutor(max_workers=max_workers)
                futures: list[Future] = []

                calibration_data_list = []

                # 处理每张图片
                for j, one_line_paths_list in enumerate(dataset_path_list):
                    full_img_path = one_line_paths_list[idx]

                    # 使用OpenCV读取图片
                    img = cv2.imread(full_img_path)
                    if img is None:
                        print(f"Warning: Could not read image {full_img_path}")
                        continue

                    # 等比缩放 + 居中填充 + BGR转RGB + 布局/类型转换 (复用 utils.letterbox_image)
                    img_float = letterbox_image(
                        img,
                        (width, height),
                        output_format=set_input_order,
                        output_dtype='float32',
                    )

                    # 调试窗口: 显示处理后的图像 (RGB -> BGR 保持颜色正确)
                    display_img = img_float.squeeze().astype(np.uint8)
                    if set_input_order == 'nhwc':
                        display_img = display_img
                    else:
                        display_img = np.transpose(display_img, (1, 2, 0))
                    cv2.imshow("padded_image", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                    # 生成输出文件名
                    base_name = os.path.splitext(os.path.basename(full_img_path))[0]
                    output_path = os.path.join(output_dir, f"{base_name}.raw")

                    calibration_data_list.append(output_path)

                    # 提交文件保存任务到线程池
                    future = Threadpool_to_file.submit(to_file_thread, img_float, output_path)
                    futures.append(future)

                    if len(futures) >= max_workers:
                        print(f"Processed a batch of {max_workers} images")
                        concurrent.futures.wait(futures, timeout=2)

                        for i in range(len(futures)):
                            future = futures.pop(0)
                            if future.done() is False:
                                futures.append(future)

                cv2.destroyAllWindows()
                # 等待所有文件保存任务完成
                concurrent.futures.wait(futures)
                Threadpool_to_file.shutdown(wait=True)

                file_list = [os.path.abspath(file_path) for file_path in calibration_data_list]

                calibration_files.append(file_list)


            # 创建当前输入的校准数据索引文件
            calibration_data_index = self.tmp_dir / f"calibration_data.txt"
            with open(str(calibration_data_index), 'w') as f:
                # 使用zip_longest处理不等长列表，空值用空字符串填充
                for row in zip_longest(*calibration_files, fillvalue=''):
                    # 过滤掉空字符串，但保留位置（这样列对齐）
                    formatted_row = ' '.join(item if item else '' for item in row)
                    f.write(formatted_row + '\n')
                print(f'{calibration_data_index} created listing {len(calibration_files)} columns.')

            print("Calibration data generation completed successfully!")

            self.file_or_dir_to_clean.append(calibration_data_index)
            for file_list in calibration_files:
                self.file_or_dir_to_clean.extend(file_list)

            return calibration_data_index

        except Exception as e:
            print(f"Error generating calibration data: {str(e)}")
            return None

    def quantize_model(self, dlc_model_path:str, calibration_data_index_path:str) -> str|None:
        dlc_model_file = Path(str(dlc_model_path))
        input_list_str = str(calibration_data_index_path)
        quantized_dlc_model_path = dlc_model_file.parent / f"{dlc_model_file.stem}_quantized.dlc"

        if not dlc_model_file.exists():
            print(f"Error: DLC model not found at {dlc_model_file}")
            return False
        

        quantize_args = f'--weights_bitwidth {self.weights_bitwidth}'
        quantize_args += f' --act_bitwidth {self.act_bitwidth} '
        quantize_args += f' --bias_bitwidth {self.bias_bitwidth}'
        quantize_args += f' --use_per_channel_quantization'
        quantize_args += f' --param_quantizer_calibration {self.param_quant_method}'
        quantize_args += f' --act_quantizer_calibration {self.act_quant_method}'
        quantize_args += f" --param_quantizer_schema {self.param_quant_schema}"
        quantize_args += f" --act_quantizer_schema {self.act_quant_schema}"
        if self.use_cle_algorithm:
            quantize_args += " --use_cle_algorithm"

        extra_args = f'{quantize_args} --target_backend HTP'
        
        command = f"qairt-quantizer --input_dlc {dlc_model_path} --input_list {input_list_str} --output_dlc {quantized_dlc_model_path} {extra_args}"

        with temporary_chdir(self.tmp_dir):
            return_code = self.run_subprocess(command)

        self.file_or_dir_to_clean.append(self.tmp_dir / 'output')

        if return_code == 0:
            self.file_or_dir_to_clean.append(quantized_dlc_model_path)
            print("Model quantization completed successfully!")
            return quantized_dlc_model_path
        else:
            print("Error during model quantization.")
            return None

    def write_config_file(self, dlc_model_path:str) -> str:
        dlc_model_file = Path(str(dlc_model_path))

        config_backend_path = dlc_model_file.parent / "config_backend.json"
        config_file_path = dlc_model_file.parent / "config_file.json"

        architecture_config = self.architecture_dict[self.target_platform]
        graph_name = Path(dlc_model_path).stem # self.tmp_onnx_path.stem 

        # 创建配置字典
        config_backend = {
            "graphs": [
                {
                    "graph_names": [graph_name],
                    "vtcm_mb": 0
                }
            ],
            "devices": [
                {
                    "dsp_arch": architecture_config["dsp_arch"],
                    "soc_id": architecture_config["soc_id"],
                }
            ]
        }


        config_file = {
            "backend_extensions": {
                "shared_library_path": "libQnnHtpNetRunExtensions.so",
                "config_file_path": str(config_backend_path)
            }
        }

        # 将配置写入JSON文件
        with open(str(config_backend_path), 'w') as f:
            json.dump(config_backend, f, indent=4)  # indent=4 使输出格式化，更易读

        with open(str(config_file_path), 'w') as f:
            json.dump(config_file, f, indent=4)

        self.file_or_dir_to_clean.append(str(config_backend_path))
        self.file_or_dir_to_clean.append(str(config_file_path))
        
        print(f"Config file created at: {config_backend_path}")
        return config_file_path

    def generate_context_binary_model(self, quantized_dlc_model_path:str, config_path:str):

        command = f"qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHtp.so --config_file {config_path}"
        command += f" --dlc_path {quantized_dlc_model_path} --output_dir {self.qnn_model_path.parent} --binary_file {self.qnn_model_path.stem}"

        return_code = self.run_subprocess(command)

        if return_code == 0:
            print("Context binary generation completed successfully!")
            return True
        else:
            print("Error during context binary generation.")
            return False



if __name__ == "__main__":
    onnx_path = './yolo11s.onnx'
    qnn_model_path = './yolo11s.bin'
    dataset_path = './datasets/datasets_face.txt'
    
    mean_rgb = [[0, 0, 0]]
    std_rgb = [[255, 255, 255]]

    onnx_to_qnn = OnnxToQNN(onnx_path, qnn_model_path, dataset_path)
    onnx_to_qnn.set_quantization_method(param_quant_method='percentile', act_quant_method='entropy', bitwidth='w8a8', bias_bitwidth=8)

    # 可选: 混合量化 —— 与 onnx_to_rknn.py 的 do_hybrid_quantization 一致,
    # 通过子图输入/输出张量指定区域(自动识别两者之间的节点), 可指定多个子图,
    # 子图之外仍按全局 w8a8 量化。张量名可以是节点名(自动取该节点输出张量)。
    # 1) 整数混合量化: 区域 w8a16 (权重8bit, 激活16bit), 全局默认 w8a8
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], weights_bitwidth=8, act_bitwidth=16)
    # 2) 浮点保留: 区域保持 FP16
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], float_bitwidth=16)
    # 3) 多个子图
    # onnx_to_qnn.do_hybrid_quantization([['in1', 'out1'], ['in2', 'out2']], float_bitwidth=16)

    onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.clean()
