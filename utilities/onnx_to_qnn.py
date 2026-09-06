import os
import sys
import json
import platform
import shutil
import subprocess
from pathlib import Path
from itertools import zip_longest

import numpy as np
import cv2
import onnx



current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import temporary_chdir, NumpySaver
from utils import letterbox_image, clean_files_or_dirs, read_dataset_txt_to_list, run_command
from utils import sanitize_name, parse_bitwidth, find_hybrid_subgraph_nodes
from utils import get_onnx_model_info, normalize_onnx_model, reorder_onnx_nodes_by_input, reorder_onnx_nodes_by_output






class QAIRTScript:
    """QAIRT 无扩展名 Python shebang 脚本的跨平台路径字典 + 调用前缀生成器。

    背景：qairt-converter / qairt-quantizer 等在 SDK 中是无扩展名的 Python
    shebang 脚本，位于 bin/<arch> 子目录下：
        x86_64 Windows : bin/x86_64-windows-msvc/<tool>
        x86_64 Linux   : bin/x86_64-linux-clang/<tool>
        ARM64  Linux   : bin/aarch64-oe-linux-gcc11.2/<tool>
    Windows 无法直接执行无扩展名文件，须以 `python <脚本路径>` 调用；
    Linux 由 shebang 直接执行。本类以字典集中管理各平台脚本相对路径，
    get() 按当前平台自动输出完整调用前缀，使下方执行逻辑无需关心平台差异。

    用法：
        QAIRTScript.get('qairt-converter')
        # Windows -> 'python "D:\\...\\qairt\\2.38.0.250901\\bin\\x86_64-windows-msvc\\qairt-converter"'
        # Linux   -> '"/opt/qairt/.../bin/x86_64-linux-clang/qairt-converter"'
    """

    # 工具名 -> 各平台脚本相对路径（相对 SDK 根目录）。
    # 只记录当前流程实际用到的工具，其余按需扩展。
    # 若某版本 SDK 未提供对应工具，运行时由系统自然报错，不做额外判断。
    TOOL_PATHS = {
        'qairt-converter': {
            'x86_64-windows-msvc': 'bin/x86_64-windows-msvc/qairt-converter',
            'x86_64-linux-clang': 'bin/x86_64-linux-clang/qairt-converter',
            'aarch64-oe-linux-gcc11.2': 'bin/aarch64-oe-linux-gcc11.2/qairt-converter',
        },
        'qairt-quantizer': {
            'x86_64-windows-msvc': 'bin/x86_64-windows-msvc/qairt-quantizer',
            'x86_64-linux-clang': 'bin/x86_64-linux-clang/qairt-quantizer',
            'aarch64-oe-linux-gcc11.2': 'bin/aarch64-oe-linux-gcc11.2/qairt-quantizer',
        },
    }

    # 类级 SDK 根目录（由 set_sdk_root() 设置；get() 无显式参数时使用）
    qairt_sdk_root: str | Path | None = None

    # QAIRT 工具绑定 Python ABI：QAIRT_PYTHON 默认当前解释器
    if 'QAIRT_PYTHON' not in os.environ:
        os.environ['QAIRT_PYTHON'] = sys.executable

    @classmethod
    def set_sdk_root(cls, sdk_root: str | Path) -> None:
        """
        设置类级 SDK 根目录（版本目录），之后 get() 无需再传 sdk_root。
        优先级（get 解析时）：显式传入 sdk_root > set_sdk_root 设置的 > 环境变量 QAIRT_SDK_ROOT。
        """
        cls.qairt_sdk_root = Path(sdk_root).resolve()
        print(f"[QAIRTScript] SDK root set: {cls.qairt_sdk_root}")

    @classmethod
    def current_platform_arch(cls) -> str:
        """
        返回当前平台 key：'x86_64-windows-msvc' / 'x86_64-linux-clang' / 'aarch64-oe-linux-gcc11.2'。
        Windows 统一使用 x86_64 版工具（ARM64 Windows 由高通转译执行 x86 工具，
        """

        if sys.platform.startswith('win'):
            return 'x86_64-windows-msvc'
        
        else:
            machine = platform.machine().lower()

            if machine in ('arm64', 'aarch64'):
                return 'aarch64-oe-linux-gcc11.2'
            else:
                return 'x86_64-linux-clang'

    @classmethod
    def get_tool(cls, tool_name:str) -> str:
        """按当前平台输出工具调用前缀。

        Args:
            tool_name: 工具键名，如 'qairt-converter'。
        Returns:
            str: 调用前缀，如
                Windows -> 'python "D:\\...\\bin\\x86_64-windows-msvc\\qairt-converter"'
                Linux   -> '/opt/.../bin/x86_64-linux-clang/qairt-converter'
                下方拼接参数即可得到完整命令。

        Raises:
            FileNotFoundError: 当前平台 SDK 未提供该工具，或脚本文件不存在。
        """

        platform_key = cls.current_platform_arch()
        rel_path = cls.TOOL_PATHS[tool_name].get(platform_key)
        if rel_path is None:
            raise FileNotFoundError(f'{tool_name} is not provided by the QAIRT SDK on {platform_key} ')

        sdk_root = cls.qairt_sdk_root
        if sdk_root is None:
            sdk_root = os.environ.get('QAIRT_SDK_ROOT')

        script_path = Path(sdk_root) / rel_path

        # 原生可执行文件（.exe / ELF，如 qnn-net-run）直接执行，不加 python 前缀；
        # python 脚本（shebang，如 qairt-converter/qairt-quantizer）按平台解释器调用。
        if sys.platform.startswith('win'):
            if script_path.suffix.lower() == '.exe':
                return f'"{script_path}"'
            else:
                # Windows：无扩展名脚本须经 python 解释器调用（绑定 Python ABI）。
                # 解释器优先级：QAIRT_PYTHON 环境变量 > PATH 中的 python。
                python_exe = sys.executable
                return f'"{python_exe}" "{script_path}"'

        else:
            return f'"{script_path}"'


class QnnHybridQuantGen:
    # v1.0.0 quantization_overrides 对称性取值（字段 is_sym: bool）：
    # v1 schema 每项只有 {name, bw, dtype(int/float), is_sym}，只能表达 signed
    # 的对称/非对称；'unsignedsymmetric'（unsigned）无法在 v1 overrides 中表达，
    # 需要走 AIMET/QDQ 或 v2.0.0（output_dtype=uint8）路线，这里直接拒绝。
    SUPPORTED_SCHEMAS = ('asymmetric', 'symmetric')

    def __init__(self, custom_hybrid:list[list[str]], bitwidth:str='w8a16',
                 bias_bitwidth:int|None=None, float_bitwidth:int|None=None,
                 param_quant_schema:str|None=None, act_quant_schema:str|None=None):
        """
        生成 QAIRT v1.0.0 混合量化 overrides（只标记位宽/类型，scale 由后续
        qairt-quantizer 校准计算）。未指定的可选参数不写入 encoding，避免污染：
        - 对称性未指定(schema=None)：encoding 项不写 is_sym 字段；
        - 整数模式 bias_bitwidth 未指定(None)：不把 bias 写入 param_encodings，
          交由全局量化(默认 --bias_bitwidth 8)处理。

        Args:
            custom_hybrid: [输入张量, 输出张量] 对列表（节点名亦可，取输出张量）。
            bitwidth: 量化位宽字符串 'w<W>a<A>'，默认 'w8a16'。整数模式由它决定
                区域内权重(W)/激活(A) 位宽（可选 'w4a8'/'w4a16'/'w8a8'/'w8a16'/
                'w16a16'），内部 parse_bitwidth 解析并做范围校验。
            bias_bitwidth: 整数模式下区域内偏置位宽，可选 8/32。默认 None 表示不
                override 区域内 bias（不写入 encoding）。
            float_bitwidth: 若设置(16/32)，区域保持浮点(FP16/FP32)，dtype 用
                'float'，忽略 bitwidth/bias_bitwidth/对称性；区域内 bias 随子图
                保持浮点一并写入。
            param_quant_schema: 区域内权重对称性，'asymmetric'/'symmetric'。
                None(默认) 表示不写 is_sym；仅在整数模式生效。
            act_quant_schema: 区域内激活对称性，同上。
        """
        if float_bitwidth is not None:
            if float_bitwidth not in (16, 32):
                raise ValueError('float_bitwidth must be 16 or 32')
            dtype = 'float'
            weights_bitwidth = float_bitwidth
            act_bitwidth = float_bitwidth
            # float 模式：子图整体浮点，bias 随之写入 float；对称性无意义
            bias_bitwidth = float_bitwidth
            param_quant_schema = None
            act_quant_schema = None
        else:
            dtype = 'int'
            # bitwidth 字符串 'w<W>a<A>' 在内部解析（复用 utils.parse_bitwidth）。
            try:
                weights_bitwidth, act_bitwidth = parse_bitwidth(bitwidth)
            except AttributeError:
                raise ValueError(
                    f"bitwidth must be in 'w<W>a<A>' format like 'w8a16', got {bitwidth!r}") from None
            if weights_bitwidth not in (4, 8, 16):
                raise ValueError('weights_bitwidth must be 4, 8 or 16')
            if act_bitwidth not in (8, 16):
                raise ValueError('act_bitwidth must be 8 or 16')
            if bias_bitwidth is not None and bias_bitwidth not in (8, 32):
                raise ValueError('bias_bitwidth must be 8 or 32 (or None to not override bias)')

        if not isinstance(custom_hybrid, list) or not custom_hybrid:
            raise ValueError('custom_hybrid must be a non-empty list of [input_tensor, output_tensor] pairs')

        if param_quant_schema is not None and param_quant_schema not in self.SUPPORTED_SCHEMAS:
            raise ValueError(f"param_quant_schema must be one of {self.SUPPORTED_SCHEMAS} or None, got {param_quant_schema!r}; "
                             "'unsignedsymmetric' is not representable in v1.0.0 overrides (use AIMET/QDQ path)")
        if act_quant_schema is not None and act_quant_schema not in self.SUPPORTED_SCHEMAS:
            raise ValueError(f"act_quant_schema must be one of {self.SUPPORTED_SCHEMAS} or None, got {act_quant_schema!r}; "
                             "'unsignedsymmetric' is not representable in v1.0.0 overrides (use AIMET/QDQ path)")

        self.hybrid_quantization = {
            "custom_hybrid": custom_hybrid,
            "dtype": dtype,
            "weights_bitwidth": weights_bitwidth,
            "act_bitwidth": act_bitwidth,
            "bias_bitwidth": bias_bitwidth,
            "param_quant_schema": param_quant_schema,
            "act_quant_schema": act_quant_schema,
        }
        print(f"[QnnHybridQuantGen] Hybrid quantization is set: {self.hybrid_quantization}")

    def generate_hybrid_quantization_overrides(self, tmp_onnx_path:str) -> str|None:
        """
        根据子图的输入/输出张量生成 QAIRT v1.0.0 混合量化的 quantization_overrides JSON。
        每个 [输入张量, 输出张量] 对之间的所有节点按 do_hybrid_quantization 指定的
        精度标记，转换器会在子图边界自动插入 Convert 节点。

        v1.0.0 schema（QAIRT converter 按 'version' 分支解析）：
            activation_encodings / param_encodings 为 list，每项含
            "name" / "bw"(位宽) / "dtype"("int"|"float")，仅当显式指定对称性时
            才附加 "is_sym"(bool)，未指定则省略该键以免污染。
            QAIRT 校验只强制 "bw"，允许缺 scale/offset（scale 留待 qairt-quantizer
            用校准数据计算），故只指定位宽的混合量化语义成立。
            整数模式未指定 bias_bitwidth 时，区域内 bias 不写入（交由全局量化）。

        Returns:
            str | None: overrides JSON 文件路径；失败返回 None。
        """
        try:
            model = onnx.load_model(tmp_onnx_path)
        except Exception as e:
            print(f"[QnnHybridQuantGen] Error loading ONNX model for hybrid quantization: {e}")
            return None

        nodes = list(model.graph.node)

        # 识别子图节点：每个 [输入张量, 输出张量] 对 -> 输入下游 ∩ 输出上游的节点并集
        # （复用共享的 utils.find_hybrid_subgraph_nodes，与 AIMET 路径的搜索逻辑一致）
        try:
            middle = find_hybrid_subgraph_nodes(model, self.hybrid_quantization["custom_hybrid"])
        except ValueError as e:
            print(f"[QnnHybridQuantGen] Error: {e}")
            return None

        if not middle:
            print("[QnnHybridQuantGen] Error: no nodes selected for hybrid quantization")
            return None

        hq = self.hybrid_quantization
        dtype = hq["dtype"]
        act_bw = hq["act_bitwidth"]
        weight_bw = hq["weights_bitwidth"]
        bias_bw = hq["bias_bitwidth"]          # float 模式=float bw；int 模式 None(不写) 或 8/32
        act_schema = hq["act_quant_schema"]    # None 或 'asymmetric'/'symmetric'（int 模式）
        param_schema = hq["param_quant_schema"]

        # v1.0.0 encoding 项 {name, bw, dtype}；仅当对称性显式指定(int 模式 schema
        # 非 None)才附加 is_sym；float 模式无对称性概念，不写。
        def make_encoding(name:str, bw:int, schema:str|None) -> dict:
            enc = {"name": name, "bw": bw, "dtype": dtype}
            if dtype == 'int' and schema is not None:
                enc["is_sym"] = (schema == 'symmetric')
            return enc

        initializer_names = {init.name for init in model.graph.initializer}
        activation_encodings: list[dict] = []
        param_encodings: list[dict] = []
        seen_activations: set[str] = set()
        seen_params: set[str] = set()

        # 识别 bias: 作为 Conv/ConvTranspose/Gemm 第3个输入(index 2)的 initializer
        bias_names = set()
        for n in nodes:
            if n.op_type in ('Conv', 'ConvTranspose', 'Gemm') and len(n.input) > 2:
                bias_names.add(n.input[2])

        for idx in middle:
            n = nodes[idx]
            for out in n.output:
                if out and out not in seen_activations:
                    seen_activations.add(out)
                    activation_encodings.append(make_encoding(out, act_bw, act_schema))
            for inp in n.input:
                if inp in initializer_names and inp not in seen_params:
                    is_bias = inp in bias_names
                    # int 模式未指定 bias_bitwidth(None) 时不 override bias，
                    # 由全局量化处理（避免默认位宽污染 encoding）。
                    if is_bias and dtype == 'int' and bias_bw is None:
                        continue
                    seen_params.add(inp)
                    param_bw = bias_bw if is_bias else weight_bw
                    param_encodings.append(make_encoding(inp, param_bw, param_schema))

        overrides = {
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
            "version": "1.0.0",
        }

        overrides_path = Path(tmp_onnx_path).parent / "quantization_overrides.json"
        with open(str(overrides_path), 'w') as f:
            json.dump(overrides, f, indent=4)

        print(f"[QnnHybridQuantGen] Hybrid quantization overrides generated: {len(middle)} nodes, "
              f"{len(activation_encodings)} activation tensors, {len(param_encodings)} param tensors -> {overrides_path}")
        return str(overrides_path)


class QnnAimetConnector:
    """AIMET 量化路径连接器，把 AIMET PTQ 接入 OnnxToQNN 转换流程。

    由 OnnxToQNN.set_use_aimet() 创建。量化路径：AIMET 校准量化 ->
    导出 QDQ ONNX -> qairt-converter 直接转量化 DLC（跳过 qairt-quantizer，
    编码由 AIMET 决定，与 QNN calibration 解耦）-> context binary，
    可选精度分析。OnnxToQNN.convert() 检测到 self.aimet_connector 后
    委托其 convert() 执行。
    """

    def __init__(self, converter:'OnnxToQNN', config_file:str|None=None):
        """创建连接器并保存 AIMET 量化配置。

        Args:
            converter: 父级 OnnxToQNN 实例。convert() 通过它回调
                convert_onnx_model / write_config_file / generate_context_binary_model，
                并读取校准数据、混合量化与精度分析器状态。
            config_file: AIMET quantsim_config 路径或别名（'default'/'htp_v68'...）。
                传 htp 版本时在此解析为对应内置配置的绝对路径；None 时由
                AimetOnnxQuantizer 按 'default' 处理。
        """

        self.converter = converter

        # 传入 htp 版本（如 'htp_v68'/'htp_v73'）时，在此加载对应版本的内置
        # quantsim_config，得到其绝对路径。对称性等后续由 AimetOnnxQuantizer.
        # _build_sim 基于该内置配置改写（defaults 级）后应用，绝不改动算子级配置。
        if config_file is not None:
            from onnx_aimet_quant import AimetQuantsimConfig
            self.config_file = AimetQuantsimConfig.resolve_path(config_file)
        else:
            self.config_file = None
        if self.config_file is not None:
            print(f"[QnnAimetConnector] load quantsim_config: {self.config_file}")

    def get_quantization_method(self, quant_method:str, bitwidth:str, param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric', use_cle_algorithm:bool=False):
        """设置 AIMET 量化方案、位宽与权重/激活对称性（convert() 前调用）。

        Args:
            quant_method: AIMET 方案（'min_max'/'tf_enhanced'/'percentile' 及别名）。
            bitwidth: 全局位宽 'w<W>a<A>'，如 'w8a8'。
            param_quant_schema: 权重对称性（'asymmetric'/'symmetric'/'unsignedsymmetric'）。
                默认 'symmetric'。
            act_quant_schema: 激活对称性（'asymmetric'/'symmetric'/'unsignedsymmetric'）。
                默认 'asymmetric'。
            use_cle_algorithm: 是否启用 Cross-Layer Equalization（CLE）。默认 False。
        """
        if param_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('param_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')
        
        if act_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('act_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric')

        self.quant_method = quant_method
        self.bitwidth = bitwidth
        self.param_quant_schema = param_quant_schema
        self.act_quant_schema = act_quant_schema
        self.use_cle_algorithm = use_cle_algorithm

        print(f"[QnnAimetConnector] Enabled AIMET 2.x quantization path (scheme={self.quant_method}, {bitwidth}")

    def current_hybrid_config(self) -> tuple:
        """实时读取 converter.hybrid_quantizer 的混合量化配置（与调用顺序无关）。

        Returns:
            (hybrid_subgraphs, hybrid_bitwidth, hybrid_float_bitwidth)：
            子图 [in, out] 对列表、子图位宽 'w<W>a<A>'、浮点保留位宽
            （16/32 表示子图保持 FP16/FP32，None 表示按 bitwidth 量化）。
            未设置混合量化时返回 (None, 'w8a16', None)。
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
        """执行 AIMET 量化路径，由 OnnxToQNN.convert() 委托调用。

        流程：AIMET 校准量化 -> 导出 QDQ ONNX（+ encodings）-> 按 graph.input
        顺序重排 QDQ 输入链 -> qairt-converter 转量化 DLC -> 写 config ->
        生成 context binary -> 可选精度分析（FP32 golden DLC vs 量化 DLC）。

        Args:
            onnx_model_info: ONNX 模型信息（get_onnx_model_info 产出，含 inputs 等）。
            mean_rgb / std_rgb: 每个输入的 RGB 归一化参数（精度分析用）。
            set_input_order: 输入布局，'nhwc'/'nchw'（校准 .raw 的布局解释）。
        """
        converter = self.converter
        tmp_onnx_path = converter.tmp_onnx_path
        qdq_model_path = converter.tmp_work_dir / f"{tmp_onnx_path.stem}_qdq.onnx"

        # 1.
        from onnx_aimet_quant import AimetOnnxQuantizer
        quantizer = AimetOnnxQuantizer(str(tmp_onnx_path), qdq_model_path, converter.dataset_path, self.config_file)

        # 2.
        quantizer.set_quantization_method(self.quant_method, self.bitwidth, self.param_quant_schema, self.act_quant_schema, self.use_cle_algorithm)

        # 混合量化实时读取 converter.hybrid_quantizer（兼容 do_hybrid_quantization
        # 2.5
        hybrid_subgraphs, hybrid_bitwidth, hybrid_float_bitwidth = self.current_hybrid_config()
        if hybrid_subgraphs:
            quantizer.do_hybrid_quantization(hybrid_subgraphs, hybrid_bitwidth, hybrid_float_bitwidth)

        # 2.75
        if converter.custom_calibration_data_path is not None:
            quantizer.use_custom_calibration_data(converter.custom_calibration_data_path)
        elif converter.dataset_path is None:
            # AIMET 量化必须有真实校准数据：随机 dummy 无法反映真实激活分布，直接报错
            raise ValueError("AIMET quantization requires calibration data: provide dataset_path or "
                "call use_custom_calibration_data(path) before convert().")

        # 3.
        model_info = get_onnx_model_info(str(tmp_onnx_path))
        input_shapes = [tuple(d for d in i['shape']) for i in model_info['inputs']]
        mean = [[0] * input_shape[1] for input_shape in input_shapes]
        std = [[1] * input_shape[1] for input_shape in input_shapes]

        qdq_path, enc_path = quantizer.convert(mean, std, normalize_model=False, export_encodings=True)

        converter.file_or_dir_to_clean.append(qdq_path)
        converter.file_or_dir_to_clean.append(enc_path)

        # AIMET 导出的 QDQ 输入消费链排列顺序可能与 graph.input 不一致（多输入模型常见），
        # 会导致 qairt-converter 推导出的 DLC 输入顺序错位（运行时按 graph.input 顺序
        # 喂数据时 NPU 输入错乱）。按 graph.input 顺序重排 QDQ 输入链后再转换。
        # 3.5
        self.reorder_qdq_input_chains_by_graph_order(qdq_path)

        # 转换 QDQ ONNX -> 量化 DLC（AIMET 编码直接进入 DLC，无需 qairt-quantizer）
        # is_quantized=True 时 convert_onnx_model 自动给 DLC 命名加 _quantized 后缀
        # 4.
        dlc_model_path = converter.convert_onnx_model(onnx_model_info, set_input_order, input_network_path=qdq_path, is_quantized=True)
        if dlc_model_path is None:
            exit(1)

        # 5.
        config_path = converter.write_config_file(dlc_model_path)

        # 6.
        converter.generate_context_binary_model(dlc_model_path, config_path)

        # 7. 精度分析（可选）
        if converter.accuracy_analyzer is not None and dlc_model_path is not None:
            # 精度分析 golden：纯浮点 DLC
            golden_dlc_path = converter.convert_onnx_model(onnx_model_info, set_input_order,
                                                           input_network_path=str(tmp_onnx_path),
                                                           output_dlc_name=f"{tmp_onnx_path.stem}_golden")

            converter.accuracy_analyzer.set_model_info(onnx_model_info, set_input_order)
            return_code = converter.accuracy_analyzer.accuracy_analysis(
                golden_dlc_path=golden_dlc_path,
                target_dlc_path=dlc_model_path,
                mean_rgb=mean_rgb, std_rgb=std_rgb,
            )

            if return_code == 0:
                print("[QnnAimetConnector] Accuracy analysis completed successfully.")
            else:
                print("[QnnAimetConnector] Accuracy analysis failed.")

    @staticmethod
    def reorder_qdq_input_chains_by_graph_order(qdq_model_path:str) -> str:
        """按 graph.input 顺序重排 QDQ 输入消费链（原地保存，仅调节点顺序、不改数值）。

        AIMET 导出的 QDQ 输入消费链排列顺序可能与 graph.input 不一致（多输入模型
        常见），会导致 qairt-converter 推导出的 DLC 输入顺序错位（运行时按
        graph.input 顺序喂数据时 NPU 输入错乱）。单输入模型直接跳过。

        Args:
            qdq_model_path: QDQ ONNX 路径（原地重排保存）。

        Returns:
            str: 重排后的 QDQ ONNX 路径（与入参相同）。
        """
        model = onnx.load_model(qdq_model_path)
        graph = model.graph
        input_names = [i.name for i in graph.input]
        init_names = {init.name for init in graph.initializer}
        real_inputs = [n for n in input_names if n not in init_names]
        if len(real_inputs) < 2:
            return qdq_model_path

        model = reorder_onnx_nodes_by_input(model, 5, aggressive=True)
        model = reorder_onnx_nodes_by_output(model, 10, aggressive=True)

        onnx.checker.check_model(model, full_check=True)
        onnx.save_model(model, qdq_model_path)
        print(f"[QnnAimetConnector] Reordered QDQ input chains to match graph input order: {real_inputs} (aggressive)")
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


        current_dir = Path(__file__).resolve().parent # 获取当前文件所在目录的绝对路径
        qairt_path = current_dir / 'qairt'

        # 版本号间隔较大，依赖目录名排序即可：字典序最大的即最新版本
        version_dir = max(
            (d for d in qairt_path.iterdir() if d.is_dir()),
            key=lambda d: d.name,
        )
        print(f"[OnnxToQNN] Selected latest QAIRT SDK version dir: {version_dir}")
        self.qnn_sdk_dir = version_dir
        QAIRTScript.set_sdk_root(self.qnn_sdk_dir)

        self.tmp_dir = current_dir / 'tmp' # 构建tmp根目录的绝对路径

        # 与 onnx_to_rknn.py 一致：每个模型一个独立工作子目录，避免多模型
        # 转换时中间产物（dlc/calibration/output 等）在同一目录下互相污染。
        sanitize_model_name = sanitize_name(self.model_path.stem)
        self.tmp_work_dir = self.tmp_dir / f"{sanitize_model_name}_to_qnn"
        self.tmp_work_dir.mkdir(parents=True, exist_ok=True)

        tmp_onnx_path = self.tmp_work_dir / sanitize_model_name
        self.tmp_onnx_path = tmp_onnx_path.with_suffix('.onnx')

        self.quantize_args:dict[str, str|int|bool|None] = {
            'param_quant_method': 'min-max',
            'act_quant_method': 'min-max',
            'bitwidth': 'w8a8',
            'bias_bitwidth': None,
            'param_quant_schema': None,
            'act_quant_schema': None,
            'use_cle_algorithm': False
        }

        self.custom_calibration_data_path = None

        self.file_or_dir_to_clean = []
        self.accuracy_analyzer = None
        self.hybrid_quantizer = None
        self.aimet_connector = None

    def set_quantization_method(self, param_quant_method:str='min-max', act_quant_method:str='min-max', bitwidth:str='w8a8',
                                bias_bitwidth:int|None=None, param_quant_schema:str|None=None, act_quant_schema:str|None=None,
                                use_cle_algorithm:bool=False):
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

            bias_bitwidth (int | None): Bitwidth for bias quantization.
                - Available options: 8, 32.
                - Default: None (qairt-quantizer default).

            param_quant_schema (str | None): Parameter(weight) quantization schema.
                - Available options: 'asymmetric', 'symmetric', 'unsignedsymmetric'.
                - Default: None (qairt-quantizer default).

            act_quant_schema (str | None): Activation quantization schema.
                - Available options: 'asymmetric', 'symmetric', 'unsignedsymmetric'.
                - Default: None (qairt-quantizer default).
                
            use_cle_algorithm (bool): Whether to use the Cross Layer Equalization algorithm for quantization.
        """

        if param_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('param_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if act_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('act_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')
        
        if bias_bitwidth is not None and bias_bitwidth not in [8, 32]:
            raise ValueError('bias_bitwidth must be 8 or 32 (or None to not pass --bias_bitwidth)')

        if param_quant_schema is not None and param_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('param_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric or None')
        
        if act_quant_schema is not None and act_quant_schema not in ['asymmetric', 'symmetric', 'unsignedsymmetric']:
            raise ValueError('act_quant_schema must be one of asymmetric, symmetric, unsignedsymmetric or None')
        
        self.quantize_args['param_quant_method'] = param_quant_method
        self.quantize_args['act_quant_method'] = act_quant_method
        self.quantize_args['bitwidth'] = bitwidth
        self.quantize_args['bias_bitwidth'] = bias_bitwidth
        self.quantize_args['param_quant_schema'] = param_quant_schema
        self.quantize_args['act_quant_schema'] = act_quant_schema
        self.quantize_args['use_cle_algorithm'] = use_cle_algorithm

        print(f"[OnnxToQNN] Quantization method set to: quant_method: {self.quantize_args}")

    def use_custom_calibration_data(self, custom_calibration_data_path:str|None=None):
        """
        Args:
            custom_calibration_data_path (str | None): Path to a text file containing the custom calibration dataset.
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

        if custom_calibration_data_path is None:
            self.custom_calibration_data_path = None
        else:
            self.custom_calibration_data_path = Path(custom_calibration_data_path).resolve()

        print(f"[OnnxToQNN] Custom calibration dataset path set to: {self.custom_calibration_data_path}")

    def do_hybrid_quantization(self, custom_hybrid:list[list[str, str]], bitwidth:str='w8a16',
                               bias_bitwidth:int|None=None, float_bitwidth:int|None=None,
                               param_quant_schema:str|None=None, act_quant_schema:str|None=None):
        """
        设置混合量化(与 onnx_to_rknn.py 的 do_hybrid_quantization 一致)：
        通过子图的输入张量与输出张量指定区域，自动识别两者之间的所有节点，
        对这些节点使用指定精度，子图之外的节点仍按 set_quantization_method
        的全局设置(默认 w8a8)量化为 INT8。

        两种模式(二选一)：
        1. 整数混合量化(默认)：bitwidth 指定区域内权重/激活位宽。
           例如全局 w8a8、区域 w8a16：bitwidth='w8a16'。
        2. 浮点保留：float_bitwidth 指定区域保持浮点精度，16=FP16，32=FP32。

        说明：生成的 overrides 使用 QAIRT v1.0.0 schema（activation_encodings /
        param_encodings 为 list + name，位宽键 bw），只标记用户显式指定的内容：
        schema 未指定时不写 is_sym；整数模式 bias_bitwidth 未指定时不把 bias
        写入 overrides（交由全局量化），避免默认值污染 encoding。真实 scale 由
        后续 qairt-quantizer 校准计算。

        Args:
            custom_hybrid (list[list[str]]): 每个内层列表为 [输入张量名, 输出张量名]，
                表示一个混合量化子图：输入张量与输出张量之间的所有节点被选中。
                可传入多个子图，例如 [[in1, out1], [in2, out2]]。
                张量名也可以是节点名(自动取该节点的输出张量作为边界)。

            bitwidth (str): Quantization bitwidth configuration in format 'w<W>a<A>', 
                where W is weight bitwidth and A is activation bitwidth.
                - Available options: 'w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16'.
                - Default: 'w8a16'.

            bias_bitwidth (int | None): 整数模式下区域内的偏置位宽，可选 8/32。
                None(默认) 表示不 override 区域内 bias（不写入 encoding，交由
                全局量化处理）。仅在整数模式生效。
            float_bitwidth (int | None): 若设置(16/32)，区域保持浮点(FP16/FP32)，
                忽略 bitwidth/bias_bitwidth/对称性。默认 None 表示整数混合量化。
            param_quant_schema (str | None): 区域内权重对称性，'asymmetric'/
                'symmetric'。None(默认) 表示不写 is_sym 字段。
                v1.0.0 overrides 仅支持这两者（unsignedsymmetric 需走 AIMET/QDQ 路径）。
                仅在整数模式生效。
            act_quant_schema (str | None): 区域内激活对称性，'asymmetric'/'symmetric'。
                None(默认) 表示不写 is_sym 字段。同上。
        """

        self.hybrid_quantizer = QnnHybridQuantGen(custom_hybrid, bitwidth, bias_bitwidth,
                                                  float_bitwidth, param_quant_schema, act_quant_schema)

    def set_use_aimet(self, quant_method:str='tf_enhanced', bitwidth:str="w8a8", param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric',
                      use_cle_algorithm:bool=False):
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

        self.aimet_connector = QnnAimetConnector(self, config_file)
        self.aimet_connector.get_quantization_method(quant_method, bitwidth, param_quant_schema, act_quant_schema, use_cle_algorithm)

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]|None=None):
        """
        Args:
            accuracy_analysis_picture_list (list[str], optional): A list of image paths required for model accuracy analysis. 
                - Each element in the list should be a path to an image. 
                - For models with a single input, provide a single image path. 
                - For models with multiple inputs, provide multiple image paths. Example: ['/home/xxx/1.jpg', '/home/xxx/2.jpg']
                - Defaults to None.
        """
        from accuracy_debugger import QnnAccuracyDebugger
        self.accuracy_analyzer = QnnAccuracyDebugger(self.tmp_work_dir, self.tmp_onnx_path, accuracy_analysis_picture_list)

        print(f"[OnnxToQNN] Accuracy analysis data list set to: {accuracy_analysis_picture_list}")


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

        # 1. 初始化 QAIRT 环境（source envsetup.sh），失败则终止转换
        ret = self.run_env_script()
        if not ret:
            print("[OnnxToQNN] Error: QAIRT environment initialization failed (run_env_script failed), aborting conversion.")
            exit(1)

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

        print(f"[OnnxToQNN] Model info: {onnx_model_info}")

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
        if self.dataset_path is not None and self.custom_calibration_data_path is None:
            calibration_data_index_path = self.generate_calibration_data(onnx_model_info, set_input_order)
        else:
            calibration_data_index_path = self.custom_calibration_data_path

        # 7. 量化：区分「未请求量化」与「请求了量化但校准数据生成失败」
        if calibration_data_index_path is not None:
            quantized_dlc_model_path = self.quantize_model(dlc_model_path, calibration_data_index_path)
        elif self.dataset_path is None and self.custom_calibration_data_path is None:
            # 未提供校准数据集：跳过量化，直接输出未量化 DLC
            print("[OnnxToQNN] No calibration data provided, skipping quantization, outputting unquantized DLC.")
            quantized_dlc_model_path = dlc_model_path
        else:
            # 提供了数据集但校准数据生成失败，禁止静默回退到未量化模型
            print("[OnnxToQNN] Error: calibration data generation failed, cannot quantize, aborting conversion.")
            exit(1)

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

            quantized_dlc_model_path = Path(quantized_dlc_model_path)

            # QAIRTAccuracyDebugger：直接用两个 DLC（FP32 golden + 量化 target）对比
            self.accuracy_analyzer.set_model_info(onnx_model_info, set_input_order)
            
            return_code = self.accuracy_analyzer.accuracy_analysis(
                golden_dlc_path=golden_dlc_path,
                target_dlc_path=quantized_dlc_model_path,
                mean_rgb=mean_rgb, std_rgb=std_rgb,
            )

            if return_code == 0:
                print("[OnnxToQNN] Accuracy analysis completed successfully.")
            else:
                print("[OnnxToQNN] Accuracy analysis failed.")

    def clean(self):
        """清理本次转换产生的临时文件/目录: file_or_dir_to_clean 中登记的项"""
        clean_files_or_dirs(self.file_or_dir_to_clean)

        if self.accuracy_analyzer:
            self.accuracy_analyzer.clean()


    def run_env_script(self):
        """按平台加载 QAIRT SDK 环境：Windows -> envsetup.ps1，Linux/Unix -> envsetup.sh。
        Windows 分支（x86_64 / ARM64）：dot-source bin/envsetup.ps1 并解析环境变量。
        Returns:
            bool: 环境加载成功返回 True，失败返回 False。
        """
        if sys.platform.startswith('win'):
            # ---------------- Windows ----------------
            envsetup_script = self.qnn_sdk_dir / 'bin' / 'envsetup.ps1'
            powershell_exe = shutil.which('powershell') or shutil.which('pwsh')

            if platform.machine().lower() in ('amd64', 'x86_64'):
                arch = 'X86_64'
            else:
                arch = 'ARM64'

            # 同一 PowerShell 会话 dot-source envsetup.ps1（显式 -arch 避开 WMI 检测），
            # 再导出全部环境变量；[INFO]/[WARN] 日志在 ===QAIRT_ENV_START=== 之前，解析时跳过
            command = (
                f"& {{ . '{envsetup_script}' -arch {arch}; "
                f"Write-Output '===QAIRT_ENV_START==='; "
                f"Get-ChildItem Env: | ForEach-Object {{ \"{{0}}={{1}}\" -f $_.Name, $_.Value }} }}"
            )
            proc = subprocess.Popen([powershell_exe, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, encoding='utf-8', errors='replace')
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"[OnnxToQNN] Error executing envsetup.ps1: {stderr}")
                return False

            # 只解析 env 导出段（标记行之后），前面的 [INFO]/[WARN] 日志忽略
            env_block = stdout.split('===QAIRT_ENV_START===', 1)
            if len(env_block) != 2:
                print(f"[OnnxToQNN] Error parsing envsetup.ps1 output: {stdout}")
                return False

            env_to_parse = env_block[1]

        else:
            # ---------------- Linux/Unix ----------------
            envsetup_script = self.qnn_sdk_dir / 'bin/envsetup.sh'
            command = f"source '{envsetup_script}' && env"
            print("[OnnxToQNN] Setting up QAIRT Linux environment...")

            # 执行脚本
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
            stdout, stderr = proc.communicate() # 获取输出

            if proc.returncode != 0:
                print(f"[OnnxToQNN] Error executing script: {stderr.decode('utf-8')}")
                return False

            env_to_parse = stdout.decode('utf-8')


        # 解析环境变量
        for line in env_to_parse.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

        # QAIRT 工具需要可写临时目录（受限环境系统 TEMP 不可写），指向模型工作子目录
        # qairt_tmp_dir = self.tmp_work_dir / 'qairt_tmp'
        # qairt_tmp_dir.mkdir(parents=True, exist_ok=True)
        # os.environ['QAIRT_TMP_DIR'] = str(qairt_tmp_dir)
        # self.file_or_dir_to_clean.append(qairt_tmp_dir)

        return True

    def modify_onnx_model(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        self.tmp_work_dir.mkdir(exist_ok=True) # 确保模型工作目录存在

        if not self.model_path.exists():
            print(f"[OnnxToQNN] Error: ONNX file not found at {self.model_path}")
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
        print(f"[OnnxToQNN] Copied ONNX file to {self.tmp_onnx_path}")

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

        # input_network
        if input_network_path is None:
            input_network_path = str(self.tmp_onnx_path)

        # output_path
        if output_dlc_name is None and is_quantized:
            output_dlc_name = f"{self.tmp_onnx_path.stem}_quantized"

        if output_dlc_name is not None:
            output_dlc_path = self.tmp_onnx_path.parent / f"{output_dlc_name}.dlc"
        else:
            output_dlc_path = self.tmp_onnx_path.with_suffix('.dlc')

        # desired_input
        layout_args = ""
        for input_info in onnx_model_info.get("inputs"): 
            input_name = input_info["name"]
            
            if set_input_order == 'nhwc': # 为每个输入添加源布局和目标布局参数
                layout_args += f' --source_model_input_layout "{input_name}" NCHW --desired_input_layout "{input_name}" NHWC'
                
            layout_args += f' --desired_input_color_encoding "{input_name}" rgb rgb'

        
        # quantization
        quant_args = ""
        if not is_quantized and not self.dataset_path and not self.custom_calibration_data_path:
            quant_args += " --float_bitwidth 16"

        if quantization_overrides_path:
            self.file_or_dir_to_clean.append(quantization_overrides_path)

            quant_args += f" --quantization_overrides {quantization_overrides_path}"

            if self.hybrid_quantizer is not None:
                hybrid_quantization_dict = self.hybrid_quantizer.hybrid_quantization
                if hybrid_quantization_dict["dtype"] == "float":
                    quant_args += f" --float_bitwidth {hybrid_quantization_dict['weights_bitwidth']}"

        extra_args = "--target_backend HTP --onnx_skip_simplification " # --onnx_summary' # --preserve_onnx_output_order

        # build command
        exe_qairt_converter = QAIRTScript.get_tool('qairt-converter')
        command = f"{exe_qairt_converter} --input_network {input_network_path} --output_path {output_dlc_path} {layout_args} {quant_args} {extra_args} "

        return_code = run_command(command, signature="[OnnxToQNN]")
        
        if return_code == 0:
            print("[OnnxToQNN] Convert onnx to qnn-dlc successful!")
            self.file_or_dir_to_clean.append(output_dlc_path)
            return output_dlc_path
        
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
                output_dir = self.tmp_work_dir / f"calibration_data_for_input{idx + 1}"
                output_dir.mkdir(parents=True, exist_ok=True)
                self.file_or_dir_to_clean.append(output_dir)

                # 获取当前输入的尺寸
                input_shape = input_info["shape"]
                if len(input_shape) != 4 or input_shape[0] != 1:
                    print(f"[OnnxToQNN] Error: Unsupported input shape for input {idx + 1}")
                    continue

                height, width = input_shape[2], input_shape[3]

                calibration_data_list = []
                save_buffer:list[tuple[np.ndarray, str]] = []

                # 处理每张图片
                for j, one_line_paths_list in enumerate(dataset_path_list):
                    full_img_path = one_line_paths_list[idx]

                    # 使用OpenCV读取图片
                    img = cv2.imread(full_img_path)
                    if img is None:
                        print(f"[OnnxToQNN] Warning: Could not read image {full_img_path}")
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

                    save_buffer.append((img_float, output_path))
                    if len(save_buffer) >= 16:
                        NumpySaver.save_numpy_array(save_buffer, ".raw")
                        save_buffer.clear()


                cv2.destroyAllWindows()
                # 等待所有文件保存任务完成
                NumpySaver.save_numpy_array(save_buffer, ".raw")
                NumpySaver.flush_writes_and_close()
                save_buffer.clear()

                file_list = [os.path.abspath(file_path) for file_path in calibration_data_list]
                calibration_files.append(file_list)


            # 创建当前输入的校准数据索引文件
            calibration_data_index = self.tmp_work_dir / f"calibration_data.txt"
            with open(str(calibration_data_index), 'w') as f:
                # 使用zip_longest处理不等长列表，空值用空字符串填充
                for row in zip_longest(*calibration_files, fillvalue=''):
                    # 过滤掉空字符串，但保留位置（这样列对齐）
                    formatted_row = ' '.join(item if item else '' for item in row)
                    f.write(formatted_row + '\n')
                print(f"[OnnxToQNN] {calibration_data_index} created listing {len(calibration_files)} columns.")

            print("[OnnxToQNN] Calibration data generation completed successfully!")

            self.file_or_dir_to_clean.append(calibration_data_index)
            for file_list in calibration_files:
                self.file_or_dir_to_clean.extend(file_list)

            return calibration_data_index

        except Exception as e:
            print(f"[OnnxToQNN] Error generating calibration data: {str(e)}")
            return None

    def quantize_model(self, dlc_model_path:str, calibration_data_index_path:str) -> str|None:
        # input_dlc
        dlc_model_file = Path(dlc_model_path)

        # input_list
        input_list_str = str(calibration_data_index_path)

        # output_dlc
        quantized_dlc_path = dlc_model_file.parent / f"{dlc_model_file.stem}_quantized.dlc"

        # quantization
        weights_bitwidth, act_bitwidth = parse_bitwidth(self.quantize_args['bitwidth'])

        quantize_args = f'--weights_bitwidth {weights_bitwidth} --act_bitwidth {act_bitwidth}'
        quantize_args += f' --param_quantizer_calibration {self.quantize_args["param_quant_method"]}'
        quantize_args += f' --act_quantizer_calibration {self.quantize_args["act_quant_method"]}'
        # 以下参数仅在显式指定(非 None)时追加，未指定使用 qairt-quantizer 默认值
        if self.quantize_args['bias_bitwidth'] is not None:
            quantize_args += f' --bias_bitwidth {self.quantize_args["bias_bitwidth"]}'
        if self.quantize_args['param_quant_schema'] is not None:
            quantize_args += f" --param_quantizer_schema {self.quantize_args['param_quant_schema']}"
        if self.quantize_args['act_quant_schema'] is not None:
            quantize_args += f" --act_quantizer_schema {self.quantize_args['act_quant_schema']}"
        quantize_args += f' --use_per_channel_quantization'
        if self.quantize_args["use_cle_algorithm"]:
            quantize_args += " --apply_algorithms cle"

        extra_args = f'--target_backend HTP'

        # build command
        exe_qairt_quantizer = QAIRTScript.get_tool('qairt-quantizer')
        command = f'{exe_qairt_quantizer} --input_dlc {dlc_model_path} --input_list {input_list_str} --output_dlc {quantized_dlc_path} {quantize_args} {extra_args}'

        with temporary_chdir(self.tmp_work_dir):
            return_code = run_command(command, signature="[OnnxToQNN]")

        self.file_or_dir_to_clean.append(self.tmp_work_dir / 'output')

        if return_code == 0:
            self.file_or_dir_to_clean.append(quantized_dlc_path)
            print("[OnnxToQNN] Model quantization completed successfully!")
            return quantized_dlc_path
        else:
            print("[OnnxToQNN] Error during model quantization.")
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


        # 平台适配：HTP 后端扩展库 Linux 为 libQnnHtpNetRunExtensions.so，
        if sys.platform.startswith('win'):
            shared_library = 'QnnHtpNetRunExtensions.dll'
        else:
            shared_library = 'libQnnHtpNetRunExtensions.so'

        config_file = {
            "backend_extensions": {
                "shared_library_path": shared_library,
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
        
        print(f"[OnnxToQNN] Config file created at: {config_backend_path}")
        return config_file_path

    def generate_context_binary_model(self, quantized_dlc_model_path:str, config_path:str) -> bool:
        # model, backend
        if sys.platform.startswith('win'):
            model_lib, backend_lib = 'QnnModelDlc.dll', 'QnnHtp.dll'
        else:
            model_lib, backend_lib = 'libQnnModelDlc.so', 'libQnnHtp.so'

        # build command
        command = f'qnn-context-binary-generator --model {model_lib} --backend {backend_lib} --config_file {config_path}'
        command += f' --dlc_path {quantized_dlc_model_path} --output_dir {self.qnn_model_path.parent} --binary_file {self.qnn_model_path.stem}'

        with temporary_chdir(self.tmp_work_dir):
            return_code = run_command(command, signature="[OnnxToQNN]")

        if return_code == 0:
            print("[OnnxToQNN] Context binary generation completed successfully!")
            return True
        else:
            print("[OnnxToQNN] Error during context binary generation.")
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
    # 生成 QAIRT v1.0.0 quantization_overrides (只标位宽/对称性, scale 由校准算)。
    # 1) 整数混合量化: 区域 w8a16 (权重8bit, 激活16bit), 全局默认 w8a8
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], bitwidth='w8a16')
    # 2) 整数混合量化 + 对称性: 区域内权重 symmetric / 激活 asymmetric (v1 overrides 仅支持这两者)
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], bitwidth='w8a16',
    #                                     param_quant_schema='symmetric', act_quant_schema='asymmetric')
    # 3) 浮点保留: 区域保持 FP16
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], float_bitwidth=16)
    # 4) 多个子图
    # onnx_to_qnn.do_hybrid_quantization([['in1', 'out1'], ['in2', 'out2']], bitwidth='w16a16',
    #                                     param_quant_schema='symmetric', act_quant_schema='asymmetric')

    onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.clean()
