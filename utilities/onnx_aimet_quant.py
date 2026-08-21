"""
基于 AIMET 2.x 的 ONNX 后训练量化（PTQ）工具。

核心流程：
    1. 构建 QuantizationSimModel（param_type / activation_type / quant_scheme）
    2. 混合精度（sim.set_tensor_precision）
    3. 校准（with aimet_onnx.compute_encodings(sim): session.run(...)）
    4. 导出 QDQ ONNX（sim.to_onnx_qdq）+ encodings JSON（sim.export）
    5. （可选）AIMET encodings -> QAIRT quantization_overrides JSON

用法 A — 独立量化（API 与 OnnxToQNN / OnnxToRKNN 对齐）：
    quantizer = AimetOnnxQuantizer('model.onnx', 'model_q.onnx', dataset_path='datasets.txt')
    quantizer.set_quantization_method(quant_method='percentile', bitwidth='w8a8')
    quantizer.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
    quantizer.clean()
    # 产出: model_q.onnx (QDQ)；encodings 默认写入 work_dir（clean 时删除）
    #       需正式保留时 convert(..., export_encodings=True)

用法 B — 接入 onnx_to_qnn：
    onnx_to_qnn.set_use_aimet(...)
    onnx_to_qnn.convert(mean_rgb, std_rgb)
    # AIMET 量化 -> QDQ ONNX -> qairt-converter 直转量化 DLC
    # （跳过 qairt-quantizer，编码由 AIMET 决定，与 QNN calibration 解耦）
"""

import os
import sys
import json
import copy
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import cv2
import onnx

import aimet_onnx
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme, qtype
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.cross_layer_equalization import equalize_model


# utils.py 共享的子图节点搜索 / 归一化烘焙（与 onnx_to_qnn 的 QAIRT overrides / modify_onnx_model 同一实现）
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
from utils import letterbox_image, parse_bitwidth, clean_files_or_dirs, read_dataset_txt_to_list
from utils import get_onnx_model_info, find_hybrid_subgraph_nodes, normalize_onnx_model


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AIMET quantsim_config：别名映射 + 路径解析
# ---------------------------------------------------------------------------


class AimetQuantsimConfig:
    """AIMET quantsim_config 别名映射与路径解析。

    把项目侧的 config_file（别名 / 文件路径 / 内置文件名）解析为 AIMET
    内置 quantsim_config 的绝对路径，供 QuantizationSimModel 加载。
    """

    # 项目 config_file 别名 -> AIMET 内置 quantsim_config 文件名（HTP 各架构 + 默认）
    HTP_CONFIG: dict[str, str] = {
        'default': 'default_config_per_channel.json',
        'htp_v66': 'htp_quantsim_config_v66.json',
        'htp_v68': 'htp_quantsim_config_v68.json',
        'htp_v69': 'htp_quantsim_config_v69.json',
        'htp_v73': 'htp_quantsim_config_v73.json',
        'htp_v75': 'htp_quantsim_config_v75.json',
        'htp_v79': 'htp_quantsim_config_v79.json',
        'htp_v81': 'htp_quantsim_config_v81.json',
    }

    # AIMET 内置 quantsim_config 所在目录
    CONFIG_DIR: Path = Path(aimet_onnx.__file__).resolve().parent / 'common' / 'quantsim_config'

    @classmethod
    def resolve_path(cls, config_file: str | None) -> str:
        """把 config_file（别名 / 文件路径 / 内置文件名）解析为 quantsim_config 绝对路径。

        解析优先级：
            1. 已存在的文件路径            -> 直接返回（自定义配置）
            2. 命中 HTP_CONFIG               -> 对应内置 HTP 配置文件
            3. 其它                       -> 在内置 quantsim_config 目录下按文件名查找
            None 按 'default' 处理。

        Args:
            config_file: 别名（'default'/'htp_v73'...）、JSON 文件路径或内置文件名；
                None 按 'default' 处理。

        Returns:
            str: quantsim_config JSON 的绝对路径。

        Raises:
            FileNotFoundError: 三种方式均未命中。
        """
        cfg = str(config_file) if config_file else 'default'

        if os.path.isfile(cfg):
            return os.path.abspath(cfg)
        
        if cfg in cls.HTP_CONFIG:
            return str(cls.CONFIG_DIR / cls.HTP_CONFIG[cfg])
        
        candidate = cls.CONFIG_DIR / cfg

        if candidate.is_file():
            return str(candidate)
        
        raise FileNotFoundError(
            f"AIMET quantsim_config not found: {config_file}. Expected an alias "
            f"({sorted(cls.HTP_CONFIG)}) or a path to a quantsim_config JSON.")

class AimetQuantSchemeConfig:
    # 项目量化校准方法 -> AIMET quant_scheme 字符串（AIMET 2.x 支持 min_max / tf_enhanced / percentile）
    QUANT_SCHEME_CONFIG = {
        'min-max': 'min_max',
        'minmax': 'min_max',
        'min_max': 'min_max',
        'tf-enhanced': 'tf_enhanced',
        'tf_enhanced': 'tf_enhanced',
        'percentile': 'percentile',
        'sequential_mse': 'sequential_mse', # Sequential MSE：触发 SeqMSE 优化；量化方案内部强制用 min_max（TF quant-scheme）
        'sequential-mse': 'sequential_mse',
    }

    def resolve_quant_scheme(cls, method: str) -> str:
        """把项目校准方法名映射为 AIMET 2.x quant_scheme 字符串。

        支持 min_max / tf_enhanced / percentile 及别名（'min-max'/'tf' 等），
        以及 'sequential_mse'（触发 SeqMSE 优化）。未知方法抛 ValueError。
        """
        key = str(method).strip().lower()
        if key not in cls.QUANT_SCHEME_CONFIG:
            raise ValueError(
                f"Unsupported quant method '{method}'. Available: {sorted(cls.QUANT_SCHEME_CONFIG)}")
        return cls.QUANT_SCHEME_CONFIG[key]


def _resolve_symmetric(schema: str) -> tuple[bool, bool]:
    """把项目量化 schema 映射为 AIMET 的 (use_symmetric_encodings, use_unsigned_symmetric)。

    - 'asymmetric'        -> (False, False)：非对称
    - 'symmetric'         -> (True,  False)：有符号对称（zero_point=0）
    - 'unsignedsymmetric' -> (True,  True)：非负对称（zero_point=0，只量非负范围）

    未知 schema 抛 ValueError。
    """
    key = str(schema).strip().lower()
    if key == 'asymmetric':
        return False, False
    if key == 'symmetric':
        return True, False
    if key == 'unsignedsymmetric':
        return True, True
    raise ValueError(
        f"Unsupported quant schema '{schema}'. Available: asymmetric, symmetric, unsignedsymmetric")


def _clean_qdq_activation_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """清洗 AIMET QDQ 模型的激活张量命名，恢复原始名（原地修改并返回 model）。

    AIMET to_onnx_qdq() 会把每个量化激活张量拆成三个名字：
        X（上游节点输出，QuantizeLinear 输入）
        X_q（QuantizeLinear 输出）
        X_updated / X_qdq（DequantizeLinear 输出，后续节点消费）
    本函数把 DQ 输出恢复为原始名 X（上游节点输出腾位改名 X__src），
    使 DLC 中间层张量名与 FP32 模型一致，便于 golden/quant 逐层精度对比。

    只改名字，不改 Q/DQ 结构、scale/zero_point，量化语义不变；
    图输入 / 图输出名保持不变（外部接口）。
    """
    graph = model.graph
    producer: dict[str, int] = {}
    for i, n in enumerate(graph.node):
        for o in n.output:
            producer[o] = i
    graph_input_names = {i.name for i in graph.input}
    graph_output_names = {o.name for o in graph.output}

    rename: dict[str, str] = {}

    for n in graph.node:
        if n.op_type != 'DequantizeLinear' or not n.output or not n.input:
            continue
        dq_out = n.output[0]
        q_in = n.input[0]
        # 只处理激活张量：ONNX 路径名含 '/'；权重/偏置（weight_qdq/bias_qdq，无 '/'）不在此列
        if '/' not in dq_out:
            continue
        # 激活 DQ 输出后缀：'X_updated'（主体）或 'X_qdq'（检测头）；其输入为 Q 输出 'X_q'
        if not (dq_out.endswith('_updated') or dq_out.endswith('_qdq')):
            continue
        if not q_in.endswith('_q'):
            continue
        q_idx = producer.get(q_in)
        if q_idx is None:
            continue
        q_node = graph.node[q_idx]
        if q_node.op_type != 'QuantizeLinear' or not q_node.input:
            continue
        orig = q_node.input[0]
        # 图边界张量名保持不变（外部接口）
        if orig in graph_input_names or orig in graph_output_names:
            continue
        src_name = orig + '__src'
        if src_name in producer:
            continue  # 腾位名冲突，跳过该张量
        rename[orig] = src_name
        rename[dq_out] = orig

    if not rename:
        return model

    def remap(name: str) -> str:
        return rename.get(name, name)

    for n in graph.node:
        n.input[:] = [remap(i) for i in n.input]
        n.output[:] = [remap(o) for o in n.output]
    for vi in graph.value_info:
        vi.name = remap(vi.name)
    for t in graph.initializer:
        t.name = remap(t.name)
    for t in graph.sparse_initializer:
        t.values.name = remap(t.values.name)

    print(f"[AIMET] QDQ activation names cleaned: {len(rename)} renames applied")
    return model


# ---------------------------------------------------------------------------
# AIMET 2.x ONNX 量化器
# ---------------------------------------------------------------------------

class AimetOnnxQuantizer:
    """基于 AIMET 2.x 的 ONNX 后训练量化器（PTQ），产出 QDQ ONNX + encodings。

    两种用法：
        A. 独立量化：set_quantization_method() -> convert()，
           数据集校准 -> 量化 -> 导出 QDQ ONNX（+ encodings）。
        B. 接入 onnx_to_qnn：由 QnnAimetConnector 调用 quantize()/export()，
           产出的 QDQ 交给 qairt-converter 直接转量化 DLC。

    能力：
        - 对称性：权重/bias 经 quantsim_config 配置表（defaults 级）应用；
          激活经 _apply_quant_schema 逐量化器应用（可独立设符号）。
        - 子图混合精度：do_hybrid_quantization()。
        - 精度分析：set_do_accuracy_analysis()，convert() 内做 FP32 vs QDQ 对比。
    """

    def __init__(self, model_path:str, quantized_model_path:str|None, dataset_path:str|None = None,
                 config_file: str | None = None, fold_batch_norms:bool = True):
        """创建量化器，保存模型路径、校准数据与 quantsim_config 配置。

        Args:
            model_path: 输入 ONNX 路径（FP32）。
            quantized_model_path: QDQ ONNX 输出路径；None 时默认
                {model_path 同目录}/<stem>_qdq.onnx。
            dataset_path: 校准图片列表 txt（每行一个样本，多输入空格分隔）；
                None（默认）时改用自定义校准集（use_custom_alibration_data）。
            config_file: AIMET quantsim_config 路径或别名（'default'/'htp_v73'...）；
                None 时用 'default'。
            fold_batch_norms: 量化前是否先做 BatchNorm 折叠。默认 True。
        """

        # 临时工作目录：默认 utilities/tmp onnx_to_qnn 的 utilities/tmp 一致）
        self.work_dir = Path(__file__).resolve().parent / 'tmp'
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(model_path).resolve()

        if quantized_model_path is not None:
            self.quantized_model_path = Path(quantized_model_path).resolve()
        else:
            self.quantized_model_path = self.work_dir / f"{self.model_path.stem}_qdq.onnx"

        # dataset_path 可为 None：此时默认走自定义校准集（use_custom_alibration_data）
        self.dataset_path = Path(dataset_path).resolve() if dataset_path is not None else None

        # 位宽 / 偏置位宽由 set_quantization_method 设置（默认 w8a8）
        self.set_quantization_method()
        # 校准方法以构造参数 quant_scheme 为准

        self.config_file = config_file
        self.fold_batch_norms = fold_batch_norms

        # 与 OnnxToQNN / OnnxToRKNN 对齐的配置项
        self.accuracy_analysis_picture_list:list|None = None

        # 与 OnnxToQNN.clean() 一致：用列表记录本次量化产生的临时文件/目录，
        self.file_or_dir_to_clean:list = []

        # 子图混合量化配置（[input_tensor, output_tensor] 列表 + 位宽）
        self.hybrid_subgraphs: list | None = None
        self.hybrid_weights_bitwidth = 8
        self.hybrid_act_bitwidth = 16
        self.hybrid_float_bitwidth: int | None = None

        # 自定义校准数据（.raw/.npy 张量数据集 txt；use_custom_alibration_data 设置）
        self.custom_alibration_data_path: Path | None = None
        self.custom_data_tensor_order = "nhwc"

        # 模型输入/输出信息（get_onnx_model_info 结果；convert() 开始即加载，
        # 各方法（校准 / 精度分析等）均可访问）
        self.model_info: dict | None = None

        # AIMET 2.x QuantizationSimModel（quantize() 后有效）
        self.sim: QuantizationSimModel | None = None
        self._model: onnx.ModelProto | None = None


    # 配置
    def set_quantization_method(self, quant_method:str='tf_enhanced', bitwidth:str='w8a8',
                                param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric',
                                use_cle_algorithm:bool=False):
        """设置量化方案、位宽与权重/激活对称性（convert() 前调用）。

        bias 固定以 int32 导出，无需配置。

        Args:
            quant_method: AIMET 方案，'min_max'/'tf_enhanced'/'percentile'，
                也接受别名（'min-max'/'minmax'/'tf'/'tf-enhanced'）。
                传 'sequential_mse'（或 'sequential-mse'）启用 Sequential MSE：
                逐层搜索并冻结最优权重编码（候选数默认 20，内部强制 min_max 方案）。
            bitwidth: 位宽 'w<W>a<A>'，如 'w8a8'/'w8a16'。
            param_quant_schema: 权重对称性，'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'symmetric'（AIMET/HTP 惯例，权重对称）。
            act_quant_schema: 激活对称性，'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'asymmetric'。
            use_cle_algorithm: 是否启用 Cross-Layer Equalization（CLE）；启用时由
                equalize_model 一并执行 HighBiasFold 偏置修正，无需单独传参。默认 False。
        """
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')

        for name, schema in (('param_quant_schema', param_quant_schema),
                             ('act_quant_schema', act_quant_schema)):
            if schema not in ('asymmetric', 'symmetric', 'unsignedsymmetric'):
                raise ValueError(
                    f"{name} must be one of asymmetric, symmetric, unsignedsymmetric")

        self.quant_scheme = AimetQuantSchemeConfig.resolve_quant_scheme(quant_method)

        self.weights_bitwidth, self.act_bitwidth = parse_bitwidth(bitwidth)

        self.param_quant_schema = param_quant_schema
        self.act_quant_schema = act_quant_schema

        self.use_cle_algorithm = use_cle_algorithm

        # Sequential MSE 由 quant_method='sequential_mse'/'sequential-mse' 触发
        self.use_seq_mse = self.quant_scheme == "sequential_mse"

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list: list[str] | None = None):
        """开启精度分析：convert() 末尾对给定图片做 FP32 vs QDQ 输出对比。

        Args:
            accuracy_analysis_picture_list: 图片路径列表；None 关闭精度分析。
        """
        if accuracy_analysis_picture_list is not None:
            self.accuracy_analysis_picture_list = accuracy_analysis_picture_list
        else:
            self.accuracy_analysis_picture_list = None

    def use_custom_alibration_data(self, custom_alibration_data_path:str|None=None, dataset_tensor_order:str="nhwc"):
        """改用自定义张量校准数据（.raw / .npy，非图片），与 OnnxToQNN 行为一致。

        Args:
            custom_alibration_data_path: 校准数据 txt（每行一个样本，多输入文件路径
                空格分隔；相对路径基于 txt 目录）。每个文件为模型单个输入的张量：
                .raw —— float32 裸二进制（无 shape 头，按模型输入 shape 解释）；
                .npy —— numpy 数组文件（自带 shape）。数据需按模型输入 shape 预处理。
                None 恢复为图片数据集（dataset_path）。
            dataset_tensor_order: 自定义数据的张量布局 'nhwc'/'nchw'。默认 'nhwc'。
        """
        if custom_alibration_data_path is None:
            self.custom_alibration_data_path = None
        else:
            self.custom_alibration_data_path = Path(custom_alibration_data_path).resolve()
        self.custom_data_tensor_order = dataset_tensor_order

    def do_hybrid_quantization(self, custom_hybrid:list[list[str]], bitwidth:str="w8a16", float_bitwidth:int|None=None):
        """注册子图混合量化：[输入张量, 输出张量] 之间的节点用指定精度，其余按全局。

        Args:
            custom_hybrid: 子图列表，每项为 [in_tensor, out_tensor]（张量名或节点名）。
            bitwidth: 子图内位宽 'w<W>a<A>'，如 'w8a16'。
            float_bitwidth: 设 16/32 时子图保持 FP16/FP32（忽略 bitwidth）；None 按 bitwidth 量化。
        """
        if not isinstance(custom_hybrid, list) or not custom_hybrid:
            raise ValueError('custom_hybrid must be a non-empty list of [input, output] pairs')
        self.hybrid_subgraphs = custom_hybrid

        weights_bitwidth, act_bitwidth = parse_bitwidth(bitwidth)

        self.hybrid_weights_bitwidth = weights_bitwidth
        self.hybrid_act_bitwidth = act_bitwidth

        self.hybrid_float_bitwidth = float_bitwidth

        mode = f'float{float_bitwidth}' if float_bitwidth else f'w{weights_bitwidth}a{act_bitwidth}'
        print(f"[AIMET] hybrid quantization registered: {len(custom_hybrid)} subgraph(s), {mode}")


    # 一站式入口（与 OnnxToQNN / OnnxToRKNN 的 convert / clean 对齐）
    def convert(self, mean_rgb: list[list[int|float]]=[[0, 0, 0]], std_rgb:list[list[int|float]]=[[1, 1, 1]], 
                normalize_model:bool=True, export_encodings:bool=False) -> tuple[str, str | None]:
        """一站式量化：校准 -> AIMET 量化 -> 导出 QDQ ONNX + encodings（可选精度分析）。

        归一化策略二选一（与 OnnxToQNN 一致）：
            normalize_model=True（默认）：把归一化烘焙进模型（1x1 Conv），
                校准输入保持原始 0-255 像素；
            normalize_model=False：不烘焙模型，(x-mean)/std 在校准/验证输入上应用。

        Args:
            mean_rgb: 每个输入的 RGB 均值。默认 [[0,0,0]]（不归一化）。
            std_rgb: 每个输入的 RGB 标准差。默认 [[1,1,1]]（不归一化）。
            normalize_model: 见上。默认 True。
            export_encodings: True 输出 encodings 到 QDQ 同目录（正式保留）；
                False（默认）输出到 work_dir 临时目录（clean() 删除）。

        Returns:
            (qdq_onnx_path, encodings_path)。
        """
        # 1) 输入信息与校准输入（与 onnx_to_qnn / onnx_to_rknn 数据集格式一致）
        #    convert 开始即读取模型输入/输出信息存入 self.model_info，
        #    供校准 / 精度分析等后续步骤（及各方法）访问
        # 1. load model
        model = onnx.load_model(str(self.model_path))

        # 2. get model info
        self.model_info = get_onnx_model_info(str(self.model_path))
        if self.model_info is None:
            raise ValueError(f"Failed to read ONNX model info: {self.model_path}")
        
        print(f"Model info: {self.model_info}")
        input_names = [i['name'] for i in self.model_info['inputs']]
        input_shapes = [tuple(d for d in i['shape']) for i in self.model_info['inputs']]


        # 归一化策略（与 OnnxToQNN 一致）二选一：
        #   normalize_model=True（默认）：通过共享的 utils.normalize_onnx_model 把归一化
        #     烘焙进模型（1x1 Conv；mean=0/std=1 时内部自动跳过），校准输入保持原始 0-255 像素
        #   normalize_model=False：不烘焙模型，归一化 (x - mean) / std 改在校准/验证输入上应用
        # 3. normalize_model
        if normalize_model:
            model = normalize_onnx_model(model, mean_rgb, std_rgb)
            calibration_mean = [[0] * input_shape[1] for input_shape in input_shapes]
            calibration_std = [[1] * input_shape[1] for input_shape in input_shapes]
        else:
            calibration_mean = mean_rgb
            calibration_std = std_rgb

        # 4. prepare calibration dataset（自定义 .raw/.npy 张量，或图片数据集）
        if self.custom_alibration_data_path is not None:
            calib_iterator = custom_dataset_to_iterator(
                str(self.custom_alibration_data_path), input_names, input_shapes,
                dataset_tensor_order=self.custom_data_tensor_order)
        elif self.dataset_path is not None:
            calib_iterator = image_calibration_inputs(
                str(self.dataset_path), input_names, input_shapes,
                mean_rgb=calibration_mean, std_rgb=calibration_std)
        else:
            raise ValueError(
                "No calibration data provided: set dataset_path or call "
                "use_custom_alibration_data(...) before convert().")

        # 5. AIMET 2.x 量化（使用 normalize_model 处理后的模型）
        self.quantize(model=model, calibration_data=calib_iterator)

        # 6. 导出 QDQ ONNX + encodings
        #    encodings 默认输出到 work_dir 临时目录（clean() 时删除）；
        #    export_encodings=True 时输出到 QDQ ONNX 同目录（正式保留）
        qdq_path, enc_path = self.export(
            str(self.quantized_model_path.parent),
            self.quantized_model_path.stem,
            encoding_version='2.0.0',
            export_encodings=True,
            encodings_dir=None if export_encodings else str(self.work_dir),
        )

        # 4) 可选精度分析（FP32 vs 量化输出对比）
        if self.accuracy_analysis_picture_list is not None:
            acc_txt = self.work_dir / 'accuracy_analysis.txt'
            with open(str(acc_txt), 'w') as f:
                for line in self.accuracy_analysis_picture_list:
                    f.write(line + '\n')
            # 与校准输入一致：自定义数据用张量迭代器，图片数据用图片迭代器
            if self.custom_alibration_data_path is not None:
                acc_inputs = custom_dataset_to_iterator(
                    str(acc_txt), input_names, input_shapes,
                    dataset_tensor_order=self.custom_data_tensor_order)
            else:
                acc_inputs = image_calibration_inputs(
                    str(acc_txt), input_names, input_shapes,
                    mean_rgb=calibration_mean, std_rgb=calibration_std)
            self.compare_outputs(acc_inputs)

        return qdq_path, enc_path

    def clean(self):
        """清理 file_or_dir_to_clean 中登记的临时文件/目录（含 work_dir）。不删正式产物。"""
        clean_files_or_dirs(self.file_or_dir_to_clean)


    # ------------------------------------------------------------------
    # 量化主流程（AIMET 2.x）
    # ------------------------------------------------------------------

    def _build_quantsim_config(self, out_path: str | Path) -> str:
        """基于 AIMET 内置 quantsim_config 生成对称性改写后的配置，写入 out_path。

        通过配置表（而非逐个量化器改属性）应用权重对称性，只动 defaults 级字段：
            defaults.params.is_symmetric  —— 权重对称（param_quant_schema）
            defaults.unsigned_symmetric   —— 仅当 param/act 都非负对称时置 True
                                            （config 的 unsigned_symmetric 是全局限制，
                                             避免把有符号权重误置为 unsigned）
        激活对称性不走配置表（全局 unsigned_symmetric 会误伤有符号权重），
        由 _apply_quant_schema 逐量化器设置。
        不修改 op_type / supergroups / supergroup_pass_list / model_input /
        model_output 等算子级优化配置。

        Args:
            out_path: 输出配置路径（父目录自动创建）。

        Returns:
            str: 写入的配置路径。
        """
        base_path = AimetQuantsimConfig.resolve_path(self.config_file)
        with open(base_path) as f:
            config:dict[str, dict] = json.load(f)

        p_sym, p_uns = _resolve_symmetric(self.param_quant_schema)
        a_sym, a_uns = _resolve_symmetric(self.act_quant_schema)

        # defaults 级
        defaults_config = config.setdefault('defaults', {})
        defaults_params_config = defaults_config.setdefault('params', {})
        defaults_params_config['is_symmetric'] = str(p_sym)

        # defaults_config.setdefault('ops', {})['is_symmetric'] = str(a_sym)

        # config 的 unsigned_symmetric 为全局：仅当 param/act 都非负对称才开启，避免把
        # 有符号权重误置为 unsigned。激活的非负对称在导出时由 force_activation_as='unsigned' 处理。
        defaults_config['unsigned_symmetric'] = str(bool(p_uns and a_uns))

        # 顶层 params（weight/bias 按参数类型、全局生效，非 per-op）
        # params_config = config.setdefault('params', {})
        # params_config.setdefault('weight', {})['is_symmetric'] = str(p_sym)


        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(config, f, indent=2)
        return str(out)

    def _build_sim(self, model: onnx.ModelProto, dummy_input: dict | None = None) -> QuantizationSimModel:
        """创建 AIMET 2.x QuantizationSimModel（未校准）。

        位宽经 qtype.int 传入 param_type / activation_type；SeqMSE 模式下
        强制 quant_scheme='min_max'（AIMET 要求 TF 方案）。

        Args:
            model: 待量化的 ONNX 模型（已预处理）。
            dummy_input: 模型输入 dummy（{输入名: ndarray}）；None 由 AIMET 自动生成。

        Returns:
            QuantizationSimModel: 未校准的量化模拟模型。
        """
        # SeqMSE 在 aimet-onnx 2.36 中要求 TF quant-scheme；该版本无独立 post_training_tf，
        # 实际由 'min_max' 承担（post_training_tf 是 min_max 的别名），tf_enhanced 不满足。
        quant_scheme = self.quant_scheme
        if self.use_seq_mse:
            print(f"[AIMET] SeqMSE requires TF/min_max quant-scheme; forcing 'min_max' (was '{quant_scheme}')")
            quant_scheme = 'min_max'

        # 对称/非对称通过改写 quantsim_config（defaults 级）应用：把内置配置复制到
        # 工作目录、改好对称性后再传给 QuantizationSimModel，不再逐个量化器改属性。
        config_path = self._build_quantsim_config(self.work_dir / 'aimet_quantsim_config.json')
        self.config_file_applied = config_path

        print(f"[AIMET] quantsim_config applied: {config_path} " f"(param={self.param_quant_schema}, act={self.act_quant_schema})")

        return QuantizationSimModel(
            model,
            param_type=qtype.int(self.weights_bitwidth),
            activation_type=qtype.int(self.act_bitwidth),
            quant_scheme=quant_scheme,
            config_file=self.config_file_applied,
            dummy_input=dummy_input,
        )


    @staticmethod
    def _find_subgraph_tensors(
        model: onnx.ModelProto, custom_hybrid: list[list[str]]
    ) -> tuple[list[str], list[str]]:
        """识别子图 [输入张量, 输出张量] 之间的所有节点，返回其张量名。

        子图节点 = 每个 [in, out] 对的「输入下游 ∩ 输出上游」的并集
        （复用 utils.find_hybrid_subgraph_nodes）。

        Args:
            model: 待分析模型。
            custom_hybrid: 子图列表，每项为 [in_tensor, out_tensor]（张量名或节点名）。

        Returns:
            (子图内激活张量名, 子图内权重参数张量名)；均去重保序，权重不含 bias。

        Raises:
            ValueError: 未选中任何节点。
        """
        nodes = list(model.graph.node)

        # 子图节点：每个 [输入张量, 输出张量] 对 -> 输入下游 ∩ 输出上游的节点并集
        middle = find_hybrid_subgraph_nodes(model, custom_hybrid, warn_prefix='[AIMET] ')

        if not middle:
            raise ValueError("No nodes selected for hybrid quantization")

        initializer_names = {init.name for init in model.graph.initializer}
        bias_names = set()
        for n in nodes:
            if n.op_type in ('Conv', 'ConvTranspose', 'Gemm') and len(n.input) > 2:
                bias_names.add(n.input[2])

        act_tensors = [out for idx in middle for out in nodes[idx].output if out]
        param_tensors = [
            inp for idx in middle for inp in nodes[idx].input
            if inp in initializer_names and inp not in bias_names
        ]
        # 去重且保序
        act_tensors = list(dict.fromkeys(act_tensors))
        param_tensors = list(dict.fromkeys(param_tensors))
        return act_tensors, param_tensors

    def _apply_mixed_precision(self, model: onnx.ModelProto):
        """对 do_hybrid_quantization 注册的子图张量应用混合精度（sim.set_tensor_precision）。

        float_bitwidth 设定时子图用 qtype.float（FP16=(5,10) / FP32=(8,23)），
        否则用 qtype.int（子图位宽）。未注册子图时直接返回。

        Args:
            model: 已由 _build_sim 量化的模型（用于查找子图张量）。
        """
        precision_map: dict[str, Any] = {}

        if self.hybrid_subgraphs is not None:
            act_tensors, param_tensors = self._find_subgraph_tensors(model, self.hybrid_subgraphs)
            if self.hybrid_float_bitwidth is not None:
                # AIMET 2.x qtype.float(exp_bits, mantissa_bits) 需要指数/尾数两位参数：
                # FP16 -> (5, 10)，FP32 -> (8, 23)
                if self.hybrid_float_bitwidth == 16:
                    act_prec = qtype.float(5, 10)
                    param_prec = qtype.float(5, 10)
                elif self.hybrid_float_bitwidth == 32:
                    act_prec = qtype.float(8, 23)
                    param_prec = qtype.float(8, 23)
                else:
                    raise ValueError(
                        f"Unsupported hybrid float_bitwidth: {self.hybrid_float_bitwidth}, "
                        f"expected 16 or 32")
            else:
                act_prec = qtype.int(self.hybrid_act_bitwidth)
                param_prec = qtype.int(self.hybrid_weights_bitwidth)
            for t in act_tensors:
                precision_map[t] = act_prec
            for p in param_tensors:
                precision_map[p] = param_prec
            print(f"[AIMET] hybrid subgraph: {len(act_tensors)} act tensors, "
                  f"{len(param_tensors)} param tensors")

        if not precision_map:
            return

        for name, prec in precision_map.items():
            try:
                self.sim.set_tensor_precision(name, prec, strict=True)
            except Exception as e:
                print(f"[AIMET] Warning: set precision of '{name}' failed: {e}")
        print(f"[AIMET] mixed precision applied to {len(precision_map)} tensors")


    def _apply_quant_schema(self, sim: QuantizationSimModel):
        """对激活量化器按 act_quant_schema 设置对称性与符号（unsigned）。

        权重/bias 的对称性已由 _build_quantsim_config 经配置表应用；
        激活的非负对称（unsigned）因 config 的 unsigned_symmetric 是全局限制
        （会误伤有符号权重），只能逐量化器设置 use_symmetric_encodings /
        use_unsigned_symmetric。

        Args:
            sim: 已构建的 QuantizationSimModel（compute_encodings 前调用）。
        """
        from aimet_onnx.qc_quantize_op import QcQuantizeOp

        _, act_qs = sim.get_all_quantizers()
        act_qs: list[QcQuantizeOp]

        is_sym, is_unsigned = _resolve_symmetric(self.act_quant_schema)
        for q in act_qs:
            q.use_symmetric_encodings = is_sym
            q.use_unsigned_symmetric = is_unsigned

        print(f"[AIMET] act quant schema applied: param={self.param_quant_schema}, act={self.act_quant_schema}")


    def quantize(self, model:onnx.ModelProto, calibration_data:Iterable[dict[str, np.ndarray]],
                 forward_pass_callback: Callable | None = None, forward_pass_callback_args: Any = None) -> QuantizationSimModel:
        """AIMET 校准主流程：模型预处理 -> 构建 QuantizationSimModel -> 计算编码。

        预处理（原地）：BatchNorm 折叠（fold_batch_norms）、CLE + HighBiasFold
        （use_cle_algorithm）。随后构建 sim、应用混合精度与激活对称性，
        最后 compute_encodings 校准（SeqMSE 模式先逐层搜索并冻结最优权重编码）。

        Args:
            model: 已加载/已烘焙归一化的 ONNX 模型（会被 BN 折叠 / CLE 原地预处理）。
            calibration_data: 校准样本迭代（每项 {输入名: ndarray}）。
            forward_pass_callback: 自定义校准回调，签名 (session, args)；
                None 时用内置上下文管理器逐样本 session.run。
            forward_pass_callback_args: 传给回调的第二个参数。

        Returns:
            QuantizationSimModel: 计算完编码的量化模拟模型（存于 self.sim）。
        """

        # 记录预处理后的 FP32 模型（精度分析 / 混合精度查找子图使用）
        self._model = copy.deepcopy(model)

        if self.fold_batch_norms:
            try:
                fold_all_batch_norms_to_weight(model)
                print("[AIMET] batch norms folded into weights")
            except Exception as e:  # pragma: no cover
                print(f"[AIMET] batch norm fold skipped ({e})")

        if self.use_cle_algorithm:
            # equalize_model 内部按序执行：BN 折叠 -> CLS(跨层缩放) -> HighBiasFold(偏置修正)
            try:
                equalize_model(model)
                print("[AIMET] CLE + HighBiasFold applied (equalize_model)")
            except Exception as e:  # pragma: no cover
                print(f"[AIMET] CLE skipped ({e})")


        # 构造 dummy_input（模型带动态维度时 AIMET 需要它解析图；取首样本）
        dummy_input = None
        if isinstance(calibration_data, (list, tuple)):
            if calibration_data:
                first = calibration_data[0]
                dummy_input = {k: np.asarray(v) for k, v in first.items()}
        else:
            try:
                first = next(iter(calibration_data))
                dummy_input = {k: np.asarray(v) for k, v in first.items()}
                calibration_data = _chain(first, calibration_data)  # 恢复被消费的首元素
            except StopIteration:
                pass

        self.sim = self._build_sim(model, dummy_input)
        self._apply_mixed_precision(model)
        self._apply_quant_schema(self.sim)

        # SeqMSE：compute_encodings 前逐层搜索并冻结最优权重编码（只优化权重，
        # 激活编码仍由后面的 compute_encodings 计算）。需要把（可能为生成器的）
        # 校准样本收成可复用的 list，compute_encodings 复用同一批样本。
        if self.use_seq_mse:
            from aimet_onnx.sequential_mse.seq_mse import apply_seq_mse
            calib_list = list(calibration_data)
            calib_list_len = len(calib_list)
            candidate_num = 20

            if not calib_list_len:
                raise ValueError("SeqMSE requires at least one calibration sample.")
            elif calib_list_len > candidate_num + 2:
                # 样本较多：去掉首尾各 1 个后，均匀采样 candidate_num 个给 SeqMSE
                clipped_calib_list = calib_list[1:-1]
                clipped_len = len(clipped_calib_list)
                indices = [round(i * (clipped_len - 1) / (candidate_num - 1)) for i in range(candidate_num)]
                mse_calib_list = [clipped_calib_list[i] for i in indices]
            else:
                mse_calib_list = calib_list

            apply_seq_mse(self.sim, mse_calib_list)  # num_candidates 用默认 20
            calibration_data = iter(calib_list)

        if forward_pass_callback is not None:
            # AIMET 2.x 回调式校准
            self.sim.compute_encodings(forward_pass_callback, forward_pass_callback_args)
        else:
            # AIMET 2.x 上下文管理器校准：with compute_encodings(sim): session.run(...)
            with aimet_onnx.compute_encodings(self.sim):
                count = 0
                for sample in calibration_data:
                    self.sim.session.run(None, sample)
                    count += 1
            print(f"[AIMET] encodings computed using {count} calibration sample(s)")

        return self.sim


    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: str,
        filename_prefix: str,
        encoding_version: str = '2.0.0',
        export_encodings: bool = True,
        encodings_dir: str | None = None,
    ) -> tuple[str, str | None]:
        """导出 QDQ ONNX +（可选）encodings JSON（bias 固定以 int32 导出）。

        QDQ 模型由 sim.to_onnx_qdq() 产出（权重以量化整数常量 + DQ 存储），
        并清洗激活张量命名（恢复原始名，去掉 '_q'/'_updated'/'_qdq' 后缀）。

        Args:
            output_dir: QDQ ONNX 输出目录。
            filename_prefix: 输出文件前缀（QDQ 与 encodings 共用）。
            encoding_version: encodings 版本 '0.6.1'/'1.0.0'/'2.0.0'。
            export_encodings: 是否输出 encodings；False 时 encodings_path 为 None。
            encodings_dir: encodings 输出目录；None 时与 output_dir 相同。

        Returns:
            (qdq_onnx_path, encodings_path)。
        """
        if self.sim is None:
            raise RuntimeError("Call quantize() before export()")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if "un" in self.act_quant_schema:
            force_activation_as = "unsigned"
        else:
            force_activation_as = "signed"

        # 1) QDQ ONNX（AIMET 2.x: to_onnx_qdq）
        qdq_model = self.sim.to_onnx_qdq(
            prequantize_constants=True,
            export_int32_bias=True,
            force_activation_as=force_activation_as,
        )
        # 清洗激活张量命名：恢复原始名，避免 '_q'/'_updated'/'_qdq' 后缀污染中间层
        # （Q/DQ 结构、scale/zero_point 不变，量化语义完全一致）
        qdq_model = _clean_qdq_activation_names(qdq_model)
        qdq_path = out_dir / f"{filename_prefix}.onnx"
        onnx.save(qdq_model, str(qdq_path))

        # 2) encodings JSON（AIMET 2.x: export）—— 可选，默认输出
        if export_encodings:
            #    注意 export_model=False：模型文件只由上面的 to_onnx_qdq() 负责，
            #    否则 sim.export(export_model=True) 会向同一路径写入非 QDQ 模型覆盖它。
            enc_dir = Path(encodings_dir) if encodings_dir else out_dir
            enc_dir.mkdir(parents=True, exist_ok=True)
            self.sim.export(
                str(enc_dir),
                filename_prefix,
                export_model=False,
                encoding_version=encoding_version,
                export_int32_bias=True,
                force_activation_as=force_activation_as,
            )
            enc_path = enc_dir / f"{filename_prefix}.encodings"
            if not enc_path.exists():
                raise FileNotFoundError(f"Encodings file not produced: {enc_path}")
        else:
            enc_path = None

        qdq_nodes = sum(1 for n in qdq_model.graph.node
                        if n.op_type in ('QuantizeLinear', 'DequantizeLinear'))
        print(f"[AIMET] exported QDQ ONNX: {qdq_path} ({qdq_nodes} Q/DQ nodes)")
        if enc_path:
            print(f"[AIMET] exported encodings: {enc_path} (version={encoding_version})")
        else:
            print("[AIMET] encodings export skipped (export_encodings=False)")
        return str(qdq_path), str(enc_path)

    def export_qairt_overrides(
        self,
        encoding_1_0_path: str | None = None,
        output_path: str | None = None,
    ) -> str:
        """把 AIMET 1.0.0 encodings 转为 QAIRT quantization_overrides JSON。

        转换规则：activation 默认非对称、param 默认对称；per-channel 权重按
        name 分组为多个编码条目；scale/offset/min/max 原样透传。

        Args:
            encoding_1_0_path: AIMET 1.0.0 encodings 路径；None 时从当前 sim 导出
                （需先调用 quantize()）。
            output_path: 输出 JSON 路径；默认 {encoding 同目录}/qairt_quantization_overrides.json。

        Returns:
            str: QAIRT overrides JSON 路径。
        """
        if encoding_1_0_path is None:
            if self.sim is None:
                raise RuntimeError("Call quantize() first, or pass encoding_1_0_path")
            encoding_1_0_path = str(self.work_dir / 'encodings_1_0_0.encodings')
            self.sim.export(str(self.work_dir), 'encodings_1_0_0',
                            export_model=False, encoding_version='1.0.0')

        with open(encoding_1_0_path) as f:
            data = json.load(f)

        def to_override_item(e: dict, default_sym: str) -> dict:
            item = {
                'bitwidth': e.get('bw', e.get('bitwidth')),
                'is_symmetric': str(e.get('is_sym', e.get('is_symmetric', default_sym))),
            }
            scale = e.get('scale')
            offset = e.get('offset')
            if scale is not None:
                item['scale'] = scale[0] if isinstance(scale, list) else scale
            if offset is not None:
                item['offset'] = offset[0] if isinstance(offset, list) else offset
            if 'min' in e and e['min'] is not None:
                item['min'] = e['min']
                item['max'] = e['max']
            return item

        overrides = {
            'activation_encodings': {
                e['name']: [to_override_item(e, 'False')]
                for e in data.get('activation_encodings', [])
            },
            'param_encodings': {},
        }
        # param 按 name 分组（per-channel 同名字典多个编码条目）
        for e in data.get('param_encodings', []):
            overrides['param_encodings'].setdefault(e['name'], []).append(
                to_override_item(e, 'True'))

        if output_path is None:
            output_path = str(Path(encoding_1_0_path).with_name('qairt_quantization_overrides.json'))
        with open(output_path, 'w') as f:
            json.dump(overrides, f, indent=4)
        print(f"[AIMET] QAIRT quantization overrides written to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def compare_outputs(
        self,
        inputs: Iterable[dict[str, np.ndarray]],
        output_names: list[str] | None = None,
        num_samples: int = 8,
    ) -> tuple[float, float]:
        """用 onnxruntime 对比 FP32 与量化(QDQ)模型输出，返回 (最大绝对误差, 余弦相似度)。

        两个模型分别导出到 work_dir 临时文件后逐样本推理；QDQ 模型无法被
        onnxruntime 加载时（如混合 FP16/FP32 语义）返回 (nan, nan) 并提示。

        Args:
            inputs: 验证样本迭代（每项 {输入名: ndarray}）。
            output_names: 对比的输出张量名；None 用全部输出。
            num_samples: 最多对比样本数。

        Returns:
            (max_abs_error, cosine_similarity)：跨样本取最大绝对误差、最小余弦相似度。
        """
        if self.sim is None:
            raise RuntimeError("Call quantize() first")
        import onnxruntime as ort

        qdq_model = self.sim.to_onnx_qdq()
        fp32_model = self._model
        qdq_path = str(self.work_dir / '_compare_qdq.onnx')
        fp32_path = str(self.work_dir / '_compare_fp32.onnx')
        onnx.save(qdq_model, qdq_path)
        onnx.save(fp32_model, fp32_path)

        try:
            sess_q = ort.InferenceSession(qdq_path)
        except Exception as e:
            print(f"[AIMET] Warning: QDQ model cannot be loaded by onnxruntime: {e}")
            print("[AIMET] QDQ ONNX 仍可用于 qairt-converter；若报 FP16 类型不匹配错误，"
                  "说明模型为混合 FP16/FP32 语义，请先转为纯 FP32 模型再量化。")
            return float('nan'), float('nan')
        sess_f = ort.InferenceSession(fp32_path)
        out_names = output_names or [o.name for o in sess_f.get_outputs()]

        max_err = 0.0
        cos = 1.0
        count = 0
        for sample in inputs:
            if count >= num_samples:
                break
            out_f = sess_f.run(out_names, sample)
            out_q = sess_q.run(out_names, sample)
            for a, b in zip(out_f, out_q):
                a = np.asarray(a).astype(np.float64)
                b = np.asarray(b).astype(np.float64)
                max_err = max(max_err, float(np.max(np.abs(a - b))))
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom > 0:
                    cos = min(cos, float(np.dot(a.ravel(), b.ravel()) / denom))
            count += 1
        print(f"[AIMET] FP32 vs Quantized: max_abs_error={max_err:.6f}, cosine={cos:.6f} "
              f"({count} sample(s))")
        return max_err, cos


# ---------------------------------------------------------------------------
# 迭代器小工具
# ---------------------------------------------------------------------------

def _chain(first, it):
    """把首元素与迭代器串起来（取首样本构造 dummy_input 后恢复被消费的元素）。"""
    yield first
    yield from it


# ---------------------------------------------------------------------------
# 图片校准输入生成（独立使用时方便）
# ---------------------------------------------------------------------------

def image_calibration_inputs(dataset_path:str, input_names:list[str], input_shapes:list[tuple[int]], 
                             mean_rgb: list[list[int]], std_rgb: list[list[int]]) -> Iterator[dict[str, np.ndarray]]:
    """从图片列表 txt 生成校准输入迭代器（每样本 {输入名: np.ndarray}）。

    数据集格式与 onnx_to_qnn / onnx_to_rknn 一致：每行一个样本，多输入时各图片
    路径空格分隔，相对路径基于 txt 所在目录。
    图片预处理：letterbox 等比缩放居中填充 + BGR->RGB + NCHW + float32，
    再按 (x - mean) / std 归一化。

    Args:
        dataset_path: 图片路径 txt 文件。
        input_names: 模型输入张量名列表，长度与每行图片数一致。
        input_shapes: 每个输入的 4 维形状 (1, C, H, W)（取后两维为 HxW）。
        mean_rgb / std_rgb: 每个输入一组的 RGB 归一化参数，长度与 input_names 一致。

    Yields:
        每样本 {输入名: np.ndarray}（NCHW, float32）；某张图片读取失败时跳过该样本。
    """

    # 复用 utils.read_dataset_txt_to_list：逐行读取，按空格拆分，相对路径基于 txt
    # 所在目录解析为完整路径，返回 list[list[str]]（每行一个样本、每元素一张图片绝对路径）
    lines = read_dataset_txt_to_list(dataset_path)

    # 每输入一组 mean/std
    means = list(mean_rgb)
    stds = list(std_rgb)

    for img_paths in lines:
        sample = {}
        for idx, (name, shape) in enumerate(zip(input_names, input_shapes)):
            h, w = shape[-2:]  # (1, C, H, W)

            img = cv2.imread(img_paths[idx])
            if img is None:
                print(f"Warning: could not read {img_paths[idx]}")
                sample = {}
                break

            arr = letterbox_image(img, (w, h), output_format='nchw', output_dtype='float32')

            mean_a = np.array(means[idx], np.float32)
            std_a = np.array(stds[idx], np.float32)
            arr = (arr - mean_a.reshape(1, -1, 1, 1)) / std_a.reshape(1, -1, 1, 1)

            sample[name] = np.ascontiguousarray(arr, dtype=np.float32)
        if sample:
            yield sample

def custom_dataset_to_iterator(dataset_path:str, input_names:list[str], input_shapes:list[tuple[int]], dataset_tensor_order:str="nhwc"):
    """从自定义张量数据集 txt 生成校准输入迭代器（支持 .raw / .npy）。

    数据集格式与 image_calibration_inputs 一致：每行一个样本，多输入时各文件路径
    空格分隔，相对路径基于 txt 目录。每个文件为模型单个输入的张量：
        .raw —— 裸 float32 二进制（无 shape 头），按模型输入 shape 重新解释；
        .npy —— numpy 数组文件（自带 shape）。
    张量布局由 dataset_tensor_order 指定，统一转成模型 NCHW 布局输出。

    Args:
        dataset_path: 张量路径 txt 文件。
        input_names: 模型输入张量名列表。
        input_shapes: 每个输入的 4 维形状 (1, C, H, W)。
        dataset_tensor_order: 自定义数据的张量布局 'nhwc'/'nchw'。默认 'nhwc'。

    Yields:
        每样本 {输入名: np.ndarray}（NCHW, float32）；文件读取失败时跳过该样本。
    """
    lines = read_dataset_txt_to_list(dataset_path)
    to_nchw = dataset_tensor_order == 'nhwc'

    for file_paths in lines:
        sample = {}
        for idx, (name, shape) in enumerate(zip(input_names, input_shapes)):
            path = Path(file_paths[idx])
            try:
                if path.suffix.lower() == '.npy':
                    arr = np.load(str(path)) # numpy 数组文件：直接加载（自带 shape）

                else:
                    # 默认按 .raw 裸 float32 二进制处理：按布局 reshape
                    #   nhwc -> [N,H,W,C]；nchw -> [N,C,H,W]
                    nhwc_shape = [shape[0], shape[2], shape[3], shape[1]] if len(shape) == 4 else shape
                    arr = np.fromfile(str(path), dtype=np.float32)
                    arr = arr.reshape(nhwc_shape if to_nchw else shape)

            except Exception as e:
                print(f"Warning: could not load {path}: {e}")
                sample = {}
                break

            # 自定义布局 -> 模型 NCHW
            if to_nchw and arr.ndim == 4:
                arr = np.transpose(arr, (0, 3, 1, 2))

            sample[name] = np.ascontiguousarray(arr, dtype=np.float32)

        if sample:
            yield sample

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_standalone_quant() -> tuple[str, str]:
    """独立使用测试：AimetOnnxQuantizer 单独量化 RetinaFace_mobile。

    输入 retinaface_mobile_ModelDeploy/models_convert/onnx/ 下的 ONNX +
    datasets/datasets_face.txt 校准集，产出 QDQ ONNX 到 utilities/tmp/。

    Returns:
        (qdq_onnx_path, encodings_path)
    """
    project_dir = Path(__file__).resolve().parent.parent
    model_path = (project_dir / 'retinaface_mobile_ModelDeploy/models_convert/onnx'
                  / 'RetinaFace_mobile_[1,3,320,320].onnx')
    dataset_path = project_dir / 'datasets/datasets_face.txt'
    output_path = (project_dir / 'utilities/tmp/RetinaFace_mobile_aimet_qdq.onnx')

    # API 与 OnnxToQNN / OnnxToRKNN 一致
    quantizer = AimetOnnxQuantizer(str(model_path), str(output_path), str(dataset_path))
    quantizer.set_quantization_method(quant_method='tf_enhanced', bitwidth='w8a8', act_quant_schema="unsignedsymmetric")
    # 可选: 混合精度(子图方式, 与 onnx_to_qnn.do_hybrid_quantization 一致)
    # quantizer.do_hybrid_quantization([['/fpn/output3/output3.2/LeakyRelu_output_0', '/ssh3/Concat_output_0'],
    #                                   ['/fpn/output2/output2.2/LeakyRelu_output_0', '/ssh2/Concat_output_0'],
    #                                   ['/fpn/output1/output1.2/LeakyRelu_output_0', '/ssh1/Concat_output_0']],
    #                                  bitwidth="w8a16")
    # 可选: 精度分析（FP32 vs 量化输出对比）
    # quantizer.set_do_accuracy_analysis([str(project_dir / 'datasets/face.jpg')])

    # encodings 默认输出到 work_dir 临时目录（clean 时删除）；
    # 需要正式保留时改传 export_encodings=True
    qdq_path, enc_path = quantizer.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
    # quantizer.clean()

    return qdq_path, enc_path


if __name__ == '__main__':
    # 独立量化测试（RetinaFace_mobile）：python utilities/onnx_aimet_quant.py
    test_standalone_quant()

