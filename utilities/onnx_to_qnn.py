import os
import sys
import json
import copy
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
    """混合量化：把目标区域的精度写进 quantization_overrides，交给 QAIRT 原有两步流程。

    流程（与"先浮点转换、再统一量化"完全一致，只是中间多带一份 overrides）：
        ① qairt-converter --quantization_overrides <目标层 overrides> -> 全浮点 DLC
           overrides 以量化元数据形式随 DLC 携带（此时没给校准数据，模型本身仍是浮点）。
        ② qairt-quantizer --input_list <校准数据>（全局位宽，如 w8a8）
           量化器识别 DLC 里已注入的编码：目标层按注入位宽，其余张量按全局位宽校准补齐
           -> 混合精度 DLC。
        ③ write_config_file / generate_context_binary_model 不变。

    注入内容按位宽分两种（实测结论）：
    - w8a16 / 浮点：**只注入激活**（只写位宽/类型，0.6.1 schema）。
        bias(32bit 定点) 的 scale = weight_scale × activation_scale、PReLU coeff 等激活域
        参数、以及权重的 per-channel 量化，都由量化器按注入的激活精度重算，天然自洽。
        一旦把权重/bias 也注入（哪怕只写位宽），量化器会认为该参数"已给定"而跳过校准：
        权重丢掉 per-channel（8 条塌成 1 条）、bias 偏约 2^Δbw 倍（实测 w8a16 下 bias scale
        从 2.9e-08 变成 4.1e-10）。注入条目数还决定粒度：1 条 = per-tensor，N 条 =
        per-channel，而 per-channel 必须对称且带 offset（"Axis quantization is required to
        be symmetric..."），只写位宽会被 converter 直接拒绝。
    - w16a16：必须注入**带完整校准的 16bit 权重编码**（A16W16 要求权重对称、per-channel 每条
      带 min/max/scale/offset）。这些数值只能来自真实校准，所以在类内部调用
      quantize_w16a16_reference() 跑一遍全图 16/16 校准（带 --restrict_quantization_steps
      "-0x8000 0x7F7F"），再取目标 op 的激活/权重编码合并进 overrides。

    已知平台能力（实测 2.46，供排查参考）：
        - w8a16：v68 起可用；
        - w16a16：需要 dsp_arch >= v73（v68 报 "has incorrect Value 68, expected >= 73"）；
        - fp16：qcs6490(v68) 报 "The SocModel doesn't support FP16"，qcs8550(v73) 通过；
        - w16a8 组合在 HTP op 定义中不存在（16bit 权重必须配 16bit 激活），构造期拒绝。
    """

    # v0.6.1 quantization_overrides 的对称性取值：'is_symmetric' 为字符串 "true"/"false"
    SUPPORTED_SCHEMAS = ('asymmetric', 'symmetric')

    # A16W16 硬性要求：int16 权重范围 0x8000~0x7F7F（Hexagon 限制），必须传给 quantizer
    INT16_RESTRICT_STEPS = "-0x8000 0x7F7F"

    # 这些算子的整数语义要求输入/输出编码一致（拼接/搬运类：多个输入的整数值必须同一套
    # scale/offset 才能拼接/搬移），工具会把整链按最小位宽拉齐。区域升级时必须要求它们的
    # 所有由节点产生的输入都已升级，否则会产生不一致编码（实测 /ssh1/Concat 变成
    # "bitwidth 16 + scale=range/255"，随后 Relu 输出全 0，整网精度崩掉）。
    ENCODING_FIXED_OPS = {
        'Concat', 'Transpose', 'Reshape', 'Squeeze', 'Unsqueeze', 'Flatten', 'Expand',
        'Slice', 'Pad', 'Resize', 'Upsample', 'Gather', 'GatherElements', 'GatherND',
        'Tile', 'ChannelShuffle', 'SpaceToBatch', 'BatchToSpace', 'DepthToSpace',
        'SpaceToDepth', 'TopK', 'Shape', 'Split', 'ScatterND', 'ScatterElements',
    }

    def __init__(self, custom_hybrid:list[list[str]], bitwidth:str='w8a16',
                 float_bitwidth:int|None=None, act_quant_schema:str|None=None,
                 converter:'OnnxToQNN|None'=None):
        """
        只保存用户输入（层范围 + 精度）；scale/offset/bias 全部由 qairt-quantizer 算。

        Args:
            custom_hybrid: [输入张量, 输出张量] 对列表（节点名亦可，取输出张量）；
                [X, X] 表示只选该节点本身。
            bitwidth: 量化位宽字符串 'w<W>a<A>'，默认 'w8a16'。可选：
                'w8a16'（只注入激活，权重沿用全局 int8）/'w16a16'（内部先跑一遍 16/16
                校准，注入激活 + 完整 per-channel 权重编码）。'w16a8' 被拒绝。
            float_bitwidth: 若设置(16/32)，区域保持浮点(FP16/FP32)：激活按 float 注入，
                权重/bias 由 converter 的三件套规则自动跟随。
            act_quant_schema: 注入激活的对称性，'asymmetric'/'symmetric'；
                None(默认) 表示不写 is_symmetric。
            converter: 所属 OnnxToQNN 实例。w16a16 需要用它的 converter/quantizer 设置与
                工作目录跑一次 16/16 校准；为 None 时 w16a16 不可用。
        """
        if float_bitwidth is not None:
            if float_bitwidth not in (16, 32):
                raise ValueError('float_bitwidth must be 16 or 32')
            dtype = 'float'
            act_bitwidth = float_bitwidth
            weights_bitwidth = float_bitwidth
            act_quant_schema = None
        else:
            dtype = 'int'
            # bitwidth 字符串 'w<W>a<A>' 在内部解析（复用 utils.parse_bitwidth）。
            try:
                weights_bitwidth, act_bitwidth = parse_bitwidth(bitwidth)
            except AttributeError:
                raise ValueError(
                    f"bitwidth must be in 'w<W>a<A>' format like 'w8a16', got {bitwidth!r}") from None
            if act_bitwidth not in (8, 16):
                raise ValueError('act_bitwidth must be 8 or 16')
            # HTP op 定义里不存在「16bit 权重 + 8bit 激活」的组合（16bit 权重必须配 16bit 激活）
            if weights_bitwidth == 16 and act_bitwidth != 16:
                raise ValueError(
                    f"unsupported bitwidth {bitwidth!r}: 16-bit weights require 16-bit "
                    "activations (use 'w16a16'); HTP has no w16a8 kernel")

        if not isinstance(custom_hybrid, list) or not custom_hybrid:
            raise ValueError('custom_hybrid must be a non-empty list of [input_tensor, output_tensor] pairs')

        if act_quant_schema is not None and act_quant_schema not in self.SUPPORTED_SCHEMAS:
            raise ValueError(f"act_quant_schema must be one of {self.SUPPORTED_SCHEMAS} or None, "
                             f"got {act_quant_schema!r}")

        self.converter = converter
        self.hybrid_quantization = {
            "custom_hybrid": custom_hybrid,
            "dtype": dtype,
            "weights_bitwidth": weights_bitwidth,
            "act_bitwidth": act_bitwidth,
            "act_quant_schema": act_quant_schema,
        }
        print(f"[QnnHybridQuantGen] Hybrid quantization is set: {self.hybrid_quantization}")

    def _collect_target_tensors(self, model):
        """按 custom_hybrid 选择 ONNX 子图，返回区域内节点下标与张量名。

        选区支持两种写法：
        - [输入张量, 输出张量]：取两者之间的所有节点（find_hybrid_subgraph_nodes）；
        - [X, X]（单层写法，X 为节点名或张量名）：只选该节点本身。
          （[X, X] 不能交给下游∩上游的搜索：节点不是自身输出的消费者，交集恒为空。）

        Returns:
            (middle, activation_names, weight_names)
            middle: 区域内节点下标（升序）
            activation_names: 区域内节点产生的激活张量名
            weight_names: 区域内节点消费的权重 initializer（Conv/ConvTranspose/Gemm 的第 3 个
                输入是 bias，由量化器按精度重算，故不计入；仅 w16a16 使用该返回值）
        """
        nodes = list(model.graph.node)
        custom_hybrid = self.hybrid_quantization["custom_hybrid"]

        def resolve_single_node(name: str) -> int:
            """把 [X, X] 解析为节点下标：X 可以是节点名或该节点的任一输出张量名。"""
            for idx, n in enumerate(nodes):
                if n.name == name or name in n.output:
                    return idx
            raise ValueError(f"Tensor or node '{name}' not found in the model")

        single_node_ids: set[int] = set()
        range_pairs: list[list[str]] = []
        for pair in custom_hybrid:
            if len(pair) != 2:
                raise ValueError(f"Each custom_hybrid pair must be [input_tensor, output_tensor], got {pair}")
            if pair[0] == pair[1]:
                single_node_ids.add(resolve_single_node(pair[0]))
            else:
                range_pairs.append(list(pair))

        middle: set[int] = set(single_node_ids)
        if range_pairs:
            middle |= set(find_hybrid_subgraph_nodes(model, range_pairs))

        if not middle:
            raise ValueError("no nodes selected for hybrid quantization")
        middle = sorted(middle)

        initializers = {init.name for init in model.graph.initializer}

        bias_names = set()
        for n in nodes:
            if n.op_type in ("Conv", "ConvTranspose", "Gemm") and len(n.input) > 2:
                bias_names.add(n.input[2])

        # 生产链闭合（producer closure）：只有"所有由节点产生的输入都已升级"的节点才能升级。
        # 否则 Concat 这类多输入算子会有一个区域外(8bit)输入，工具会按文档规则把该链取
        # 最小位宽，产出"16bit 标记 + 8bit scale"的不一致编码 → 该张量及其下游全崩
        # （实测：/ssh1/Concat 变成 bw16 + scale=range/255，随后 Relu 输出全 0）。
        # graph input 与 initializer 不参与判断（整图升级时首层也能升级）。
        producer: dict[str, int] = {}
        for idx, n in enumerate(nodes):
            for out in n.output:
                if out:
                    producer[out] = idx

        upgradable: set[int] = set()
        for idx in middle:
            n = nodes[idx]
            # 只有"整数语义要求输入输出编码一致"的算子才需要闭合：它们会把整链拉齐到
            # 最小位宽，混入区域外(8bit)输入时就产生 (16bit 标记 + 8bit scale) 的不一致编码。
            # 普通计算算子(Conv/FC/Add/激活...)可以在输入侧插 Convert，不受这条限制。
            if n.op_type in self.ENCODING_FIXED_OPS:
                blocked_by = []
                for inp in n.input:
                    up = producer.get(inp)
                    if up is None:
                        continue                  # graph input / initializer：不阻塞
                    if up not in upgradable:
                        blocked_by.append(inp)
                if blocked_by:
                    continue
            upgradable.add(idx)

        # 反向收敛（consumer closure）：被升级的张量如果喂给"编码一致"算子，而那个算子还有
        # 未升级的输入（典型：Concat 拼接区域内外两侧），工具会把整链拉齐到最小位宽并产出
        # 不一致编码（实测 /ssh1/Concat 变成 bw16 + scale=range/255，随后 Relu 全 0）。
        # 因此必须把该张量的产者降回全局位宽，并迭代到稳定。
        consumers: dict[str, list[int]] = {}
        for idx, n in enumerate(nodes):
            for inp in n.input:
                consumers.setdefault(inp, []).append(idx)

        changed = True
        while changed:
            changed = False
            upgraded_tensors = set()
            for idx in upgradable:
                for out in nodes[idx].output:
                    if out:
                        upgraded_tensors.add(out)

            for idx in list(upgradable):
                downgrade = False
                for out in nodes[idx].output:
                    if not out:
                        continue
                    for consumer_idx in consumers.get(out, []):
                        consumer = nodes[consumer_idx]
                        if consumer.op_type not in self.ENCODING_FIXED_OPS:
                            continue
                        for inp in consumer.input:
                            producer_idx = producer.get(inp)
                            if producer_idx is None:
                                continue
                            if inp not in upgraded_tensors:
                                downgrade = True
                                break
                        if downgrade:
                            break
                    if downgrade:
                        break
                if downgrade:
                    upgradable.discard(idx)
                    changed = True

        activation_names: list[str] = []
        weight_names: list[str] = []
        blocked_nodes: list[str] = []
        seen_act: set[str] = set()
        seen_param: set[str] = set()

        for idx in middle:
            n = nodes[idx]
            if idx not in upgradable:
                if n.output:
                    blocked_nodes.append(n.name or n.output[0])
                else:
                    blocked_nodes.append(n.name or str(idx))
                continue
            for out in n.output:
                if out and out not in seen_act:
                    seen_act.add(out)
                    activation_names.append(out)
            for inp in n.input:
                if inp not in initializers:
                    continue
                if inp in seen_param:
                    continue
                seen_param.add(inp)
                if inp in bias_names:
                    continue
                weight_names.append(inp)

        if blocked_nodes:
            print(f"[QnnHybridQuantGen] {len(blocked_nodes)} region nodes kept at global bitwidth "
                  f"(their inputs come from outside the region): {blocked_nodes[:5]}")

        return middle, activation_names, weight_names

    def quantize_w16a16_reference(self, dlc_model_path:str, calibration_data_index_path:str,
                                  output_dlc_name:str) -> tuple[str|None, str|None]:
        """w16a16 专用量化入口：对干净浮点 DLC 跑一遍全图 16/16 校准，产出参考编码。

        只被 _build_reference_16bit_encodings 调用，参数固定不动：
        - 位宽固定 w16a16（A16W16）；
        - 权重 symmetric + 激活 asymmetric（A16W16 要求权重对称）；
        - 固定加 --restrict_quantization_steps（Hexagon int16 权重范围 0x8000~0x7F7F）。
        校准方法/bias/per-channel/CLE 沿用 converter 的全局设置。
        dump 出的编码固定写到 <输出DLC去扩展名>_encoding.json。

        Returns:
            (quantized_dlc_path, encoding_json_path)；失败返回 (None, None)。
        """
        if self.converter is None:
            print("[QnnHybridQuantGen] Error: no OnnxToQNN converter bound, cannot quantize")
            return None, None

        quantize_args = self.converter.quantize_args

        quantized_dlc_path = Path(dlc_model_path).parent / f"{output_dlc_name}.dlc"
        encoding_json_path = quantized_dlc_path.with_name(f"{quantized_dlc_path.stem}_encoding.json")

        args = '--weights_bitwidth 16 --act_bitwidth 16'
        args += f' --param_quantizer_calibration {quantize_args["param_quant_method"]}'
        args += f' --act_quantizer_calibration {quantize_args["act_quant_method"]}'
        if quantize_args["bias_bitwidth"] is not None:
            args += f' --bias_bitwidth {quantize_args["bias_bitwidth"]}'
        args += ' --param_quantizer_schema symmetric --act_quantizer_schema asymmetric'
        args += ' --use_per_channel_quantization'
        if quantize_args["use_cle_algorithm"]:
            args += ' --apply_algorithms cle'
        args += f' --restrict_quantization_steps "{self.INT16_RESTRICT_STEPS}"'
        args += ' --dump_encoding_json --target_backend HTP'

        exe_qairt_quantizer = QAIRTScript.get_tool('qairt-quantizer')
        command = (f'{exe_qairt_quantizer} --input_dlc {dlc_model_path} '
                   f'--input_list {calibration_data_index_path} '
                   f'--output_dlc {quantized_dlc_path} {args}')

        with temporary_chdir(self.converter.tmp_work_dir):
            return_code = run_command(command, signature="[QnnHybridQuantGen]")

        self.converter.file_or_dir_to_clean.append(self.converter.tmp_work_dir / 'output')

        if return_code != 0 or not encoding_json_path.exists():
            print("[QnnHybridQuantGen] Error during hybrid reference quantization.")
            return None, None

        self.converter.file_or_dir_to_clean.append(quantized_dlc_path)
        self.converter.file_or_dir_to_clean.append(encoding_json_path)
        return str(quantized_dlc_path), str(encoding_json_path)

    def _build_reference_16bit_encodings(self, onnx_model_info:dict, set_input_order:str,
                                        calibration_data_index_path, tmp_onnx_path:str) -> dict|None:
        """w16a16 用：先在干净浮点 DLC 上跑一遍全图 16/16 校准（带 int16 restrict），
        返回该精度下的完整编码（0.6.1 dict）。

        A16W16 要求 16bit 权重对称、per-channel 每条都要带 min/max/scale/offset，这些数值
        只能来自真实校准，所以这里必须多跑一遍；本函数产出的 DLC/编码都是中间产物。
        """
        if self.converter is None:
            print("[QnnHybridQuantGen] Error: w16a16 needs a bound OnnxToQNN converter")
            return None
        if not calibration_data_index_path:
            print("[QnnHybridQuantGen] Error: w16a16 needs calibration data (dataset_path / "
                  "use_custom_calibration_data)")
            return None

        # ① 干净浮点 DLC（不带 overrides），供 16/16 校准使用
        stem = Path(tmp_onnx_path).stem
        clean_float_dlc = self.converter.convert_onnx_model(
            onnx_model_info, set_input_order,
            quantization_overrides_path=None,
            output_dlc_name=f"{stem}_hybrid_float",
        )
        if clean_float_dlc is None:
            print("[QnnHybridQuantGen] Error: failed to convert clean float DLC for w16a16 reference")
            return None

        # ② 全图 16/16 校准：权重强制 symmetric，并加 A16W16 必须的 int16 范围限制
        ref_dlc, ref_encoding = self.quantize_w16a16_reference(
            clean_float_dlc, calibration_data_index_path,
            output_dlc_name=f"{stem}_hybrid_w16a16_ref",
        )
        if ref_dlc is None or ref_encoding is None:
            print("[QnnHybridQuantGen] Error: w16a16 reference quantization failed")
            return None

        print(f"[QnnHybridQuantGen] w16a16 reference calibration done: {ref_dlc}")
        with open(str(ref_encoding)) as f:
            return json.load(f)

    def generate_hybrid_quantization_overrides(self, tmp_onnx_path:str, onnx_model_info:dict|None=None,
                                               set_input_order:str='nhwc',
                                               calibration_data_index_path=None) -> str|None:
        """按目标区域生成 QAIRT 0.6.1 混合量化 overrides。

        - w8a16 / 浮点：只注入激活的位宽/类型（scale/offset/bias/权重 per-channel 全部交给
          qairt-quantizer 校准补齐，理由见类说明）；
        - w16a16：先跑一遍全图 16/16 校准（带 --restrict_quantization_steps），再把目标 op 的
          激活与权重（完整 per-channel、对称、offset=-2^15）编码一并注入，其余张量仍由全局
          位宽量化补齐。

        Returns:
            str | None: overrides JSON 文件路径；失败返回 None。
        """
        try:
            model = onnx.load_model(tmp_onnx_path)
        except Exception as e:
            print(f"[QnnHybridQuantGen] Error loading ONNX model for hybrid quantization: {e}")
            return None

        try:
            middle, activation_names, weight_names = self._collect_target_tensors(model)
        except ValueError as e:
            print(f"[QnnHybridQuantGen] Error: {e}")
            return None

        hq = self.hybrid_quantization
        dtype = hq["dtype"]
        act_bw = hq["act_bitwidth"]
        act_schema = hq["act_quant_schema"]
        use_w16 = dtype == 'int' and hq["weights_bitwidth"] == 16

        ref_encodings = None
        if use_w16:
            ref_encodings = self._build_reference_16bit_encodings(
                onnx_model_info, set_input_order, calibration_data_index_path, tmp_onnx_path)
            if ref_encodings is None:
                return None
            ref_act = ref_encodings.get("activation_encodings", {})
            ref_param = ref_encodings.get("param_encodings", {})

        activation_encodings = {}
        param_encodings = {}
        skipped = []

        for name in activation_names:
            if use_w16:
                if name in ref_act:
                    activation_encodings[name] = copy.deepcopy(ref_act[name])
                    continue
                skipped.append(name)
            encoding = {"bitwidth": act_bw, "dtype": dtype}
            # 仅当显式指定对称性(int 模式)才写 is_symmetric；float 模式无对称性概念
            if dtype == 'int' and act_schema is not None:
                if act_schema == 'symmetric':
                    encoding["is_symmetric"] = "true"
                else:
                    encoding["is_symmetric"] = "false"
            activation_encodings[name] = [encoding]

        if use_w16:
            # 权重必须用 16/16 校准里的完整 per-channel 编码（对称、带 offset），
            # 只写位宽会被 converter 拒绝或静默降级成 per-tensor
            for name in weight_names:
                if name not in ref_param:
                    skipped.append(name)
                    continue
                param_encodings[name] = copy.deepcopy(ref_param[name])

        overrides = {
            "version": "0.6.1",
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
        }

        overrides_path = Path(tmp_onnx_path).parent / "quantization_overrides.json"
        with open(str(overrides_path), 'w') as f:
            json.dump(overrides, f, indent=4)

        if use_w16:
            mode = "w16a16 (activation + full per-channel weight encodings)"
        else:
            mode = "activation-only injection"
        print(f"[QnnHybridQuantGen] Hybrid quantization overrides generated: {len(middle)} nodes, "
              f"{len(activation_encodings)} activations, {len(param_encodings)} weights [{mode}] "
              f"-> {overrides_path}")
        if skipped:
            print(f"[QnnHybridQuantGen] Warning: {len(skipped)} target tensors missing from the "
                  f"16/16 reference encodings, kept at global bitwidth (first: {skipped[:3]})")
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
                               float_bitwidth:int|None=None,
                               act_quant_schema:str|None=None):
        """
        设置混合量化(与 onnx_to_rknn.py 的 do_hybrid_quantization 一致)：
        通过子图的输入张量与输出张量指定区域，自动识别两者之间的所有节点，
        对这些节点使用指定精度，子图之外的节点仍按 set_quantization_method
        的全局设置(默认 w8a8)量化为 INT8。

        两种模式(二选一)：
        1. 整数混合量化(默认)：bitwidth 指定区域精度。
           例如全局 w8a8、区域 w8a16：bitwidth='w8a16'。
           - 'w8a16'：只注入激活位宽，权重/bias 由 qairt-quantizer 校准补齐；
           - 'w16a16'：内部先跑一遍全图 16/16 校准（带 A16W16 必须的
             --restrict_quantization_steps "-0x8000 0x7F7F"），再注入目标 op 的激活与
             完整 per-channel 权重编码（A16W16 要求权重对称）。需要 dsp_arch >= v73。
        2. 浮点保留：float_bitwidth 指定区域保持浮点精度，16=FP16，32=FP32。

        说明：生成的 overrides 使用 QAIRT 0.6.1 schema（activation_encodings /
        param_encodings 为 tensor_name -> [encoding] 的 dict）。w8a16/浮点只注入激活，
        其余张量（含权重 per-channel、bias、PReLU coeff）交给 qairt-quantizer 按注入的
        激活精度校准补齐；w16a16 额外注入 16/16 校准得到的完整权重编码。

        Args:
            custom_hybrid (list[list[str]]): 每个内层列表为 [输入张量名, 输出张量名]，
                表示一个混合量化子图：输入张量与输出张量之间的所有节点被选中。
                可传入多个子图，例如 [[in1, out1], [in2, out2]]。
                张量名也可以是节点名(自动取该节点的输出张量作为边界)；[X, X] 表示只选该节点。

            bitwidth (str): Quantization bitwidth configuration in format 'w<W>a<A>',
                where W is weight bitwidth and A is activation bitwidth.
                - Available options: 'w8a16'（推荐）/ 'w16a16'（需 v73 平台）/'w8a8'。
                - Default: 'w8a16'。

            float_bitwidth (int | None): 若设置(16/32)，区域保持浮点(FP16/FP32)。
            act_quant_schema (str | None): 注入激活的对称性，'asymmetric'/'symmetric'。
                None(默认) 表示不写 is_symmetric。
        """

        self.hybrid_quantizer = QnnHybridQuantGen(custom_hybrid, bitwidth, float_bitwidth,
                                                  act_quant_schema, converter=self)

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

        # 4.2 校准数据：混合量化的 w16a16 需要先用它跑一遍目标精度校准，故提前到转换之前
        if self.dataset_path is not None and self.custom_calibration_data_path is None:
            calibration_data_index_path = self.generate_calibration_data(onnx_model_info, set_input_order)
        else:
            calibration_data_index_path = self.custom_calibration_data_path

        # 4.3 混合量化 overrides：w16a16 时 hybrid_quantizer 内部会先跑一遍全图 16/16 校准
        if self.hybrid_quantizer is not None:
            quantization_overrides_path = self.hybrid_quantizer.generate_hybrid_quantization_overrides(
                self.tmp_onnx_path, onnx_model_info, set_input_order, calibration_data_index_path)
        else:
            quantization_overrides_path = None

        # 5.
        dlc_model_path = self.convert_onnx_model(onnx_model_info, set_input_order, quantization_overrides_path)
        if dlc_model_path is None:
            exit(1)

        # 6. (校准数据已在 4.2 生成)

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
        ret = self.generate_context_binary_model(quantized_dlc_model_path, config_path)
        if not ret:
            exit(1)

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
    # 生成 QAIRT 0.6.1 quantization_overrides: w8a16/浮点只注入激活(其余由校准补齐),
    # w16a16 额外注入 16/16 参考校准得到的完整 per-channel 权重编码。
    # 1) 整数混合量化: 区域 w8a16 (权重8bit, 激活16bit), 全局默认 w8a8
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], bitwidth='w8a16')
    # 2) 整数混合量化 + 指定激活对称性(写 is_symmetric)
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], bitwidth='w8a16',
    #                                     act_quant_schema='asymmetric')
    # 3) 浮点保留: 区域保持 FP16
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], float_bitwidth=16)
    # 4) 多个子图: 区域 w16a16 (需 dsp_arch >= v73)
    # onnx_to_qnn.do_hybrid_quantization([['in1', 'out1'], ['in2', 'out2']], bitwidth='w16a16',
    #                                     act_quant_schema='asymmetric')

    onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.clean()
