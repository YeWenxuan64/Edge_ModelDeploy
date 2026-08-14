"""
AIMET 2.x 驱动的 ONNX 后训练量化(PTQ)工具。

工作流（AIMET 2.x 新 API，非旧版弃用参数）：
    1. 构建 QuantizationSimModel —— 使用 param_type / activation_type / quant_scheme（qtype 风格）
    2. 混合精度      —— sim.set_tensor_precision(name, precision)
    3. 校准          —— with aimet_onnx.compute_encodings(sim): sim.session.run(...)
    4. 导出          —— sim.to_onnx_qdq() 产出 QDQ ONNX；sim.export() 产出 encodings JSON
    5. （可选）把 AIMET 编码转换为 QAIRT quantization_overrides JSON

两种用法：
    A. 独立量化（单独量化 onnx，API 与 OnnxToQNN / OnnxToRKNN 一致）：
        quantizer = AimetOnnxQuantizer('model.onnx', 'model_q.onnx', dataset_path='datasets.txt')
        quantizer.set_quantization_method(param_quant_method='percentile',
                                          act_quant_method='entropy', bitwidth='w8a8')
        quantizer.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
        quantizer.clean()
        产出: model_q.onnx (QDQ)；encodings 默认输出到 work_dir 临时目录
              （clean() 时删除，不污染正式产物）；需要正式保留时
              convert(..., export_encodings=True) 会额外在 QDQ 同目录产出 model_q.encodings

    B. 接入 onnx_to_qnn（配合 onnx_to_qnn.py 的 OnnxToQNN）：
        onnx_to_qnn.set_use_aimet(...)
        onnx_to_qnn.convert(mean_rgb, std_rgb)
        流程: AIMET 先把 ONNX 量化成 QDQ 模型 -> qairt-converter 直接转成量化 DLC
              （跳过 qairt-quantizer，编码由 AIMET 决定，与 QNN 的 calibration 解耦）
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
import onnx

import aimet_onnx
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme, qtype
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.cross_layer_equalization import equalize_model


# utils.py 共享的子图节点搜索 / 归一化烘焙（与 onnx_to_qnn 的 QAIRT overrides / modify_onnx_model 同一实现）
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
from utils import letterbox_image, parse_bitwidth, clean_files_or_dirs
from utils import find_hybrid_subgraph_nodes, normalize_onnx_model


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

# 项目量化校准方法 -> AIMET quant_scheme 字符串（AIMET 2.x 支持 min_max / tf_enhanced / percentile）
_QUANT_SCHEME_ALIASES = {
    'min-max': 'min_max',
    'minmax': 'min_max',
    'min_max': 'min_max',
    'tf': 'tf_enhanced',
    'tf-enhanced': 'tf_enhanced',
    'tf_enhanced': 'tf_enhanced',
    'percentile': 'percentile',
    # Sequential MSE：触发 SeqMSE 优化；量化方案内部强制用 min_max（TF quant-scheme）
    'sequential_mse': 'sequential_mse',
    'sequential-mse': 'sequential_mse',
}


def _resolve_quant_scheme(method: str) -> str:
    """把项目的校准方法名映射为 AIMET 2.x 的 quant_scheme 字符串"""
    key = str(method).strip().lower()
    if key not in _QUANT_SCHEME_ALIASES:
        raise ValueError(
            f"Unsupported quant method '{method}'. Available: {sorted(_QUANT_SCHEME_ALIASES)}")
    return _QUANT_SCHEME_ALIASES[key]


def _resolve_symmetric(schema: str) -> tuple[bool, bool]:
    """把项目的量化 schema 映射为 AIMET 的 (use_symmetric_encodings, use_unsigned_symmetric)。

    'asymmetric'      -> (False, False)：非对称
    'symmetric'       -> (True,  False)：有符号对称（zero_point=0）
    'unsignedsymmetric' -> (True,  True)：非负对称（zero_point=0，只量非负范围）
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


# ---------------------------------------------------------------------------
# AIMET quantsim_config：映射表 + 对称性应用（通过配置表，而非逐个量化器改属性）
# ---------------------------------------------------------------------------

# 项目 config_file 别名 -> AIMET 内置 quantsim_config 文件名（HTP 各架构 + 默认）
_AIMET_CONFIG_ALIASES = {
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
_AIMET_CONFIG_DIR = Path(aimet_onnx.__file__).resolve().parent / 'common' / 'quantsim_config'


def _resolve_aimet_config_path(config_file: Optional[str]) -> str:
    """把 config_file（别名 / 文件路径 / 内置文件名）解析为可加载的 quantsim_config 绝对路径。

    规则：
        None / 'default'           -> default_config_per_channel.json
        存在的文件路径              -> 直接用该路径（自定义配置）
        命中 _AIMET_CONFIG_ALIASES -> 对应内置 HTP 配置文件
        其它                        -> 当作 quantsim_config 目录下的文件名查找
    """
    cfg = str(config_file) if config_file else 'default'
    if os.path.isfile(cfg):
        return os.path.abspath(cfg)
    if cfg in _AIMET_CONFIG_ALIASES:
        return str(_AIMET_CONFIG_DIR / _AIMET_CONFIG_ALIASES[cfg])
    candidate = _AIMET_CONFIG_DIR / cfg
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"AIMET quantsim_config not found: {config_file}. Expected an alias "
        f"({sorted(_AIMET_CONFIG_ALIASES)}) or a path to a quantsim_config JSON.")


def _build_quantsim_config(base_config: Optional[str], param_schema: str, act_schema: str,
                           out_path: str | Path) -> str:
    """读取 AIMET 内置 quantsim_config，仅改写 defaults 级对称性后写入 out_path。

    通过配置表应用 param/act 的对称/非对称量化，而不是逐个量化器改
    use_symmetric_encodings / use_unsigned_symmetric。只动 defaults 级 / 顶层
    params（weight/bias 分类型，全局）字段：
        defaults.params.is_symmetric        —— 权重对称
        defaults.ops.is_symmetric           —— 激活对称（作用于全部激活）
        顶层 params.weight.is_symmetric     —— 全局权重对称（显式）
        顶层 params.bias.is_symmetric=True  —— bias 恒对称（避免 int32->uint32 导出错误）
        defaults.unsigned_symmetric         —— 仅当 param/act 都非负对称时才置 True
                                             （config 的 unsigned_symmetric 为全局限制）
    绝不修改 op_type / supergroups / supergroup_pass_list / model_input /
    model_output 等针对算子的优化配置。
    """
    base_path = _resolve_aimet_config_path(base_config)
    with open(base_path) as f:
        config:dict[str, dict] = json.load(f)

    p_sym, p_uns = _resolve_symmetric(param_schema)
    a_sym, a_uns = _resolve_symmetric(act_schema)

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


def _clean_qdq_activation_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    清洗 AIMET 2.x QDQ 模型的激活张量命名，去除 '_q' / '_updated' 后缀污染。

    AIMET to_onnx_qdq() 会把每个量化激活张量拆成三个名字：
        原始名 X（上游节点输出，QuantizeLinear 输入）
        X_q（QuantizeLinear 输出）
        X_updated / X_qdq（DequantizeLinear 输出，后续节点消费）
    （body/FPN/SSH 的激活 DQ 输出用 'X_updated'，检测头 Bbox/Class/Landmark Head
     的激活 DQ 输出用 'X_qdq'，两种都要恢复为原始名）
    本函数把 DequantizeLinear 的输出恢复为原始名 X（上游节点输出腾位改名 X__src），
    使转换后的 DLC 中间层张量名与 FP32 模型一致，便于 golden/quant 逐层精度对比。

    只改名字，不改 Q/DQ 结构、scale/zero_point，量化语义完全不变。
    图输入 / 图输出名保持不变（避免破坏外部接口）。
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
    """AIMET 2.x 后训练量化器：对 ONNX 做 PTQ，导出 QDQ ONNX + encodings。

    用法：
        A. 独立量化：set_quantization_method() -> convert()（数据集校准 -> 量化 -> 导出）。
        B. 接入 onnx_to_qnn：由 QnnAimetConnector 调用 quantize()/export()，
           QDQ 交给 qairt-converter 转量化 DLC。

    能力：
        - 对称性：权重/bias 经 quantsim_config 配置表（defaults 级）应用；激活经
          _apply_quant_schema 逐量化器应用（可独立设符号）。
        - 子图混合精度：do_hybrid_quantization()。
        - 精度分析：set_do_accuracy_analysis()，convert() 内做 FP32 vs QDQ 对比。
    """

    def __init__(
        self,
        model_path: str,
        quantized_model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        config_file: Optional[str] = None,
        fold_batch_norms: bool = True,
    ):
        """创建量化器并保存模型/校准/config 配置。

        Args:
            model_path: 输入 ONNX 路径（FP32）。
            quantized_model_path: QDQ ONNX 输出路径；None 默认
                {model_path 同目录}/<stem>_qdq.onnx。
            dataset_path: 校准图片列表 txt（每行一个样本，多输入空格分隔）；
                None 时 convert() 报错（AIMET 需真实校准数据）。
            config_file: AIMET quantsim_config 路径或别名（'default'/'htp_v73'...）；
                None 用 'default'。
            fold_batch_norms: 是否先 BatchNorm 折叠。默认 True。
        """
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        if quantized_model_path is None:
            quantized_model_path = self.model_path.with_name(f"{self.model_path.stem}_qdq.onnx")
        self.quantized_model_path = Path(quantized_model_path).resolve()

        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).resolve()
        else:
            self.dataset_path = None

        # 位宽 / 偏置位宽由 set_quantization_method 设置（默认 w8a8）
        self.set_quantization_method()
        # 校准方法以构造参数 quant_scheme 为准

        self.config_file = config_file
        self.fold_batch_norms = fold_batch_norms


        # 临时工作目录：默认 utilities/tmp onnx_to_qnn 的 utilities/tmp 一致）
        self.work_dir = Path(__file__).resolve().parent / 'tmp'
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 与 OnnxToQNN / OnnxToRKNN 对齐的配置项
        self.accuracy_analysis_picture_list: Optional[list] = None

        # 与 OnnxToQNN.clean() 一致：用列表记录本次量化产生的临时文件/目录，
        # clean() 逐项删除；work_dir（utilities/tmp/aimet）默认登记其中。
        self.file_or_dir_to_clean: list = [self.work_dir]

        # 子图混合量化配置（[input_tensor, output_tensor] 列表 + 位宽）
        self.hybrid_subgraphs: Optional[list] = None
        self.hybrid_weights_bitwidth = 8
        self.hybrid_act_bitwidth = 16
        self.hybrid_float_bitwidth: Optional[int] = None

        # AIMET 2.x QuantizationSimModel（quantize() 后有效）
        self.sim: Optional[QuantizationSimModel] = None
        self._model: Optional[onnx.ModelProto] = None

    # 配置
    def set_quantization_method(self, quant_method:str='tf_enhanced', bitwidth:str='w8a8',
                                param_quant_schema:str='symmetric', act_quant_schema:str='asymmetric',
                                use_cle_algorithm:bool=False):
        """设置量化方案、位宽与权重/激活对称性。

        设定 AIMET 量化方案（quant_method）、权重/激活位宽（bitwidth），
        以及权重/激活的对称量化方式（param_quant_schema / act_quant_schema）；
        bias 固定以 int32 导出（self.export_int32_bias 恒为 True）。

        Args:
            quant_method: AIMET 方案，'min_max'/'tf_enhanced'/'percentile'，
                也接受别名（'min-max'/'minmax'/'tf'/'tf-enhanced'）。
                传 'sequential_mse'（或 'sequential-mse'）启用 Sequential MSE
                （逐层搜索并冻结最优权重编码，候选数默认 20，内部用 min_max 方案）。
            bitwidth: 位宽 'w<W>a<A>'，如 'w8a8'/'w8a16'。
            param_quant_schema: 权重对称性，'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'symmetric'（AIMET/HTP 惯例，权重对称）。
            act_quant_schema: 激活对称性，'asymmetric'/'symmetric'/'unsignedsymmetric'。
                默认 'asymmetric'。
            use_cle_algorithm: 是否启用 Cross-Layer Equalization（CLE）；启用即由
                equalize_model 一并执行 HighBiasFold 偏置修正，无需单独传参。默认 False。
        """
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')

        for name, schema in (('param_quant_schema', param_quant_schema),
                             ('act_quant_schema', act_quant_schema)):
            if schema not in ('asymmetric', 'symmetric', 'unsignedsymmetric'):
                raise ValueError(
                    f"{name} must be one of asymmetric, symmetric, unsignedsymmetric")

        self.quant_scheme = _resolve_quant_scheme(quant_method)

        self.weights_bitwidth, self.act_bitwidth = parse_bitwidth(bitwidth)

        self.param_quant_schema = param_quant_schema
        self.act_quant_schema = act_quant_schema

        self.use_cle_algorithm = use_cle_algorithm

        # Sequential MSE 由 quant_method='sequential_mse'/'sequential-mse' 触发
        self.use_seq_mse = self.quant_scheme == "sequential_mse"

        print(f"[AIMET] quantization method set: {quant_method} {bitwidth}, "
              f"param={param_quant_schema}, act={act_quant_schema}, "
              f"cle={use_cle_algorithm}, seq_mse={self.use_seq_mse}")

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list: list[str] | None = None):
        """设置精度分析：convert() 后对给定图片做 FP32 vs QDQ 输出对比。

        Args:
            accuracy_analysis_picture_list: 图片路径列表；None 不做精度分析。
        """
        if accuracy_analysis_picture_list is not None:
            self.accuracy_analysis_picture_list = accuracy_analysis_picture_list
        else:
            self.accuracy_analysis_picture_list = None

    def do_hybrid_quantization(self, custom_hybrid: list[list[str]], bitwidth:str="w8a16", float_bitwidth: Optional[int] = None):
        """子图混合量化：对 [输入张量, 输出张量] 之间所有节点用指定精度，其余按全局。

        Args:
            custom_hybrid: 每个内层为 [in_tensor, out_tensor]（张量名或节点名）。
            bitwidth: 子图内位宽 'w<W>a<A>'，如 'w8a16'。
            float_bitwidth: 若设 16/32，子图保持 FP16/FP32（忽略 bitwidth）。
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
        """一站式量化：数据集校准 -> AIMET 量化 -> 导出 QDQ ONNX + encodings（可选精度分析）。

        Args:
            mean_rgb: 每个输入的 RGB 均值。默认 [[0,0,0]]（不归一化）。
            std_rgb: 每个输入的 RGB 标准差。默认 [[1,1,1]]（不归一化）。
            normalize_model: True（默认）把归一化烘焙进模型，校准输入为原始像素；
                False 则在校准/验证输入上应用 (x-mean)/std。
            export_encodings: True 输出 encodings 到 QDQ 同目录；False（默认）输出到
                work_dir 临时目录（clean() 删除）。

        Returns:
            (qdq_onnx_path, encodings_path)。
        """
        # 1) 输入信息与校准输入（与 onnx_to_qnn / onnx_to_rknn 数据集格式一致）
        model = onnx.load_model(str(self.model_path))
        input_names = [i.name for i in model.graph.input]
        input_shapes = [tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                        for i in model.graph.input]

        # 归一化策略（与 OnnxToQNN 一致）二选一：
        #   normalize_model=True（默认）：通过共享的 utils.normalize_onnx_model 把归一化
        #     烘焙进模型（1x1 Conv；mean=0/std=1 时内部自动跳过），校准输入保持原始 0-255 像素
        #   normalize_model=False：不烘焙模型，归一化 (x - mean) / std 改在校准/验证输入上应用
        if normalize_model:
            model = normalize_onnx_model(model, mean_rgb, std_rgb)
            calibration_mean = None
            calibration_std = None
        else:
            calibration_mean = mean_rgb
            calibration_std = std_rgb

        if self.dataset_path is not None:
            # normalize_calibration 决定校准输入是否应用 (x - mean) / std
            calib = image_calibration_inputs(
                str(self.dataset_path), input_names, input_shapes,
                mean_rgb=calibration_mean, std_rgb=calibration_std)
        else:
            raise ValueError(
                "No dataset_path provided for quantization. "
                "Pass dataset_path to AimetOnnxQuantizer(...) before convert().")

        # 2) AIMET 2.x 量化（使用 normalize_model 处理后的模型）
        self.quantize(calibration_data=calib, model=model)

        # 3) 导出 QDQ ONNX + encodings
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
            # 与校准输入一致：normalize_calibration 决定验证输入是否归一化
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

    def _load_and_prepare_model(self, model: Optional[onnx.ModelProto] = None) -> onnx.ModelProto:
        """加载模型并做可选的 BN 折叠 / CLE 预处理（均在内存中，不改源文件）。

        Args:
            model: 已加载 / 已烘焙归一化的模型；None 从 self.model_path 加载。

        Returns:
            onnx.ModelProto: 预处理后的模型。
        """
        if model is None:
            model = onnx.load_model(str(self.model_path))

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
        return model

    def _build_sim(self, model: onnx.ModelProto, dummy_input: Optional[dict] = None) -> QuantizationSimModel:
        """创建 AIMET 2.x QuantizationSimModel。

        Args:
            model: 待量化的 ONNX 模型（已预处理）。
            dummy_input: 模型输入 dummy（{输入名: ndarray}）；None 由 AIMET 自动生成。

        Returns:
            QuantizationSimModel: 未校准的量化模拟模型（含 param_type/activation_type/
                quant_scheme/config_file 配置）。
        """
        # SeqMSE 在 aimet-onnx 2.36 中要求 TF quant-scheme；该版本无独立 post_training_tf，
        # 实际由 'min_max' 承担（post_training_tf 是 min_max 的别名），tf_enhanced 不满足。
        quant_scheme = self.quant_scheme
        if self.use_seq_mse:
            print(f"[AIMET] SeqMSE requires TF/min_max quant-scheme; forcing 'min_max' (was '{quant_scheme}')")
            quant_scheme = 'min_max'

        # 对称/非对称通过改写 quantsim_config（defaults 级）应用：把内置配置复制到
        # 工作目录、改好对称性后再传给 QuantizationSimModel，不再逐个量化器改属性。
        config_path = _build_quantsim_config(
            self.config_file, self.param_quant_schema, self.act_quant_schema,
            self.work_dir / 'aimet_quantsim_config.json',
        )
        self.config_file_applied = config_path
        print(f"[AIMET] quantsim_config applied: {config_path} "
              f"(param={self.param_quant_schema}, act={self.act_quant_schema})")

        return QuantizationSimModel(
            model,
            param_type=qtype.int(self.weights_bitwidth),
            activation_type=qtype.int(self.act_bitwidth),
            quant_scheme=quant_scheme,
            config_file=config_path,
            dummy_input=dummy_input,
        )

    @staticmethod
    def _find_subgraph_tensors(
        model: onnx.ModelProto, custom_hybrid: list[list[str]]
    ) -> tuple[list[str], list[str]]:
        """识别子图 [输入张量, 输出张量] 之间所有节点，返回其张量名（复用
        utils.find_hybrid_subgraph_nodes）。

        Args:
            model: 待分析模型。
            custom_hybrid: 每个内层为 [in_tensor, out_tensor]（张量名或节点名）。

        Returns:
            (子图内激活张量名, 子图内权重参数张量名)；两者均去重保序，且权重不含 bias。
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
        """对 do_hybrid_quantization 注册的子图张量应用混合精度。

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
        """对激活量化器按 act_quant_schema 应用对称性与符号（unsigned）。

        参数（权重/bias）的对称性已由 _build_quantsim_config 通过改写 quantsim_config
        defaults 级配置应用。这里只对激活（act）量化器按 act_quant_schema 设置
        use_symmetric_encodings / use_unsigned_symmetric —— 因为 config 的
        unsigned_symmetric 是全局的（会误伤有符号权重），激活的非负对称（符号）
        仍需逐量化器设置。

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
        print(f"[AIMET] act quant schema applied: act={self.act_quant_schema}")

    def quantize(
        self,
        calibration_data: Optional[Iterable[dict[str, np.ndarray]]] = None,
        forward_pass_callback: Optional[Callable] = None,
        forward_pass_callback_args: Any = None,
        model: Optional[onnx.ModelProto] = None,
    ) -> QuantizationSimModel:
        """AIMET 2.x 校准主流程：构建 QuantizationSimModel 并计算编码。

        Args:
            calibration_data: 校准样本（{输入名: ndarray} 的迭代）。
            forward_pass_callback: 自定义校准回调，签名 (session, args)。
            forward_pass_callback_args: 传给回调的第二个参数。
            model: 预加载/已烘焙模型；None 从 self.model_path 加载。

        Returns:
            QuantizationSimModel: 计算完编码的量化模拟模型。
        """
        self._model = self._load_and_prepare_model(model)

        # 构造 dummy_input（模型带动态维度时 AIMET 需要它解析图）
        dummy_input = None
        if calibration_data is not None:
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

        self.sim = self._build_sim(self._model, dummy_input)
        self._apply_mixed_precision(self._model)
        self._apply_quant_schema(self.sim)

        # SeqMSE：compute_encodings 前逐层搜索并冻结最优权重编码（只优化权重，
        # 激活编码仍由后面的 compute_encodings 计算）。需要把（可能为生成器的）
        # 校准样本收成可复用的 list，compute_encodings 复用同一批样本。
        if self.use_seq_mse:
            if calibration_data is None:
                raise ValueError("SeqMSE requires calibration data. Pass calibration_data.")
            
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
            if calibration_data is None:
                raise ValueError(
                    "No calibration data provided. Pass calibration_data or forward_pass_callback.")
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
        encodings_dir: Optional[str] = None,
    ) -> tuple[str, str | None]:
        """导出 QDQ ONNX +（可选）encodings JSON（bias 固定以 int32 导出）。

        Args:
            output_dir: 输出目录。
            filename_prefix: 输出文件前缀。
            encoding_version: encodings 版本 '0.6.1'/'1.0.0'/'2.0.0'。
            prequantize_constants: 权重以量化整数常量（+DQ）存储。
            export_encodings: 是否输出 encodings；False 时 encodings_path 为 None。
            encodings_dir: encodings 输出目录；None 与 output_dir 相同。

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
        encoding_1_0_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """把 AIMET 1.0.0 encodings 转为 QAIRT quantization_overrides JSON。

        Args:
            encoding_1_0_path: AIMET 1.0.0 encodings 路径；None 时从当前 sim 导出。
            output_path: 输出 JSON 路径；默认 {encoding 同目录}/qairt_quantization_overrides.json。

        Returns:
            QAIRT overrides JSON 路径。
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
        output_names: Optional[list[str]] = None,
        num_samples: int = 8,
    ) -> tuple[float, float]:
        """对比 FP32 与量化(QDQ)模型输出，返回 (最大绝对误差, 余弦相似度)。

        Args:
            inputs: 验证样本（{输入名: ndarray} 的迭代）。
            output_names: 对比的输出张量名；None 用全部。
            num_samples: 最多对比样本数。

        Returns:
            (max_abs_error, cosine_similarity)。
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
    """把首元素与迭代器串起来（避免重复迭代生成器）"""
    yield first
    yield from it


# ---------------------------------------------------------------------------
# 图片校准输入生成（独立使用时方便）
# ---------------------------------------------------------------------------

def image_calibration_inputs(dataset_path:str, input_names, input_shapes, 
                             mean_rgb: Optional[list] = None, std_rgb: Optional[list] = None) -> Iterator[dict[str, np.ndarray]]:
    """
    从图片列表 txt 生成 AIMET 校准输入（每个样本一个 {输入名: np.ndarray}）。

    数据集格式与 onnx_to_qnn / onnx_to_rknn 完全一致：
        - 每行代表一个校准样本；模型有多个输入时，各输入图片路径用空格分隔
        - 相对路径基于 txt 文件所在目录（与 onnx_to_qnn.generate_calibration_data /
          onnx_to_rknn.convert 的解析逻辑一致）
    图片预处理与 onnx_to_qnn.generate_calibration_data 一致：
        letterbox 等比缩放居中填充 + BGR->RGB + 布局/类型转换，输出 float32。

    Args:
        dataset_path (str): 图片路径 txt 文件。
        input_names (str | list[str]): 模型输入张量名（单输入可传字符串）。
        input_shapes (tuple | list[tuple]): 每个输入的 (1, C, H, W)（layout='nchw'）
            或 (1, H, W, C)（layout='nhwc'）。
        mean_rgb / std_rgb (list | None): 每个输入一组的 RGB 归一化参数（与
            onnx_to_qnn / onnx_to_rknn 的 convert(mean_rgb, std_rgb) 一致）。
            None（默认）不归一化，输出原始 0-255 像素（模型内部已烘焙归一化时用）；
            非 None 时应用 (x - mean) / std 归一化（未烘焙归一化的模型用）。
        layout (str): 'nchw' 或 'nhwc'。
        border_value (tuple | None): letterbox 填充值（BGR）。None（默认）使用
            BORDER_REFLECT 镜像填充，与 onnx_to_qnn.generate_calibration_data 一致；
            指定如 (114, 114, 114) 使用常量填充（YOLO 惯例）。
    """
    import cv2  # 延迟导入

    # 兼容单输入传字符串/元组
    if isinstance(input_names, str):
        input_names = [input_names]
    if input_shapes and isinstance(input_shapes[0], int):
        input_shapes = [input_shapes]
    input_names = list(input_names)
    input_shapes = [tuple(s) for s in input_shapes]

    if len(input_names) != len(input_shapes):
        raise ValueError("input_names and input_shapes must have the same length")

    dataset_dir = Path(dataset_path).parent
    with open(dataset_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 每输入一组 mean/std（None -> 不归一化）
    if mean_rgb is None:
        means = [None] * len(input_names)
    else:
        means = list(mean_rgb) + [None] * (len(input_names) - len(mean_rgb))
    if std_rgb is None:
        stds = [None] * len(input_names)
    else:
        stds = list(std_rgb) + [None] * (len(input_names) - len(std_rgb))

    for line in lines:
        img_paths = [p for p in line.split(' ') if p]
        if len(img_paths) != len(input_names):
            print(f"Warning: line has {len(img_paths)} image(s) but model has "
                  f"{len(input_names)} input(s), skipped")
            continue
        sample = {}
        for idx, (name, shape) in enumerate(zip(input_names, input_shapes)):
            _, _, h, w = shape  # (1, C, H, W) 或 (1, H, W, C)，取后两维为 HxW
            p = Path(img_paths[idx])
            if not p.is_absolute():
                p = dataset_dir / p
            img = cv2.imread(str(p))
            if img is None:
                print(f"Warning: could not read {p}")
                sample = {}
                break

            arr = letterbox_image(img, (w, h), output_format='nchw', output_dtype='float32')

            mean = means[idx]
            std = stds[idx]
            if mean is not None or std is not None:
                mean_a = np.array(mean or [0, 0, 0], np.float32)
                std_a = np.array(std or [1, 1, 1], np.float32)
                arr = (arr - mean_a.reshape(1, -1, 1, 1)) / std_a.reshape(1, -1, 1, 1)

            sample[name] = np.ascontiguousarray(arr, dtype=np.float32)
        if sample:
            yield sample


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_standalone_quant() -> tuple[str, str]:
    """
    独立使用测试：AimetOnnxQuantizer 单独量化 RetinaFace_mobile。
    产出 QDQ ONNX + encodings 到 retinaface_mobile_ModelDeploy/models_convert/aimet/。

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

