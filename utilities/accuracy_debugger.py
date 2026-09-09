import re
import os
import sys
import csv
import json
import shutil

from pathlib import Path
from typing import Callable
from collections import defaultdict, deque

import numpy as np
import cv2
import onnx
from onnx import helper, TensorProto


current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
from utils import clean_files_or_dirs, run_command, MultThreadExetutor




# 统一"层精度"schema：内存 dict 与 CSV 共用同一套键。
# dict 形态：{层名(sanitized): {'entire_cos': float|None, 'entire_euc': float|None,
#                               'entire_mse': float|None, 'single_cos': float|None,
#                               'single_euc': float|None}}
#   - entire 侧有值 = 该层在累积(entire)分析中有结果；
#   - single 侧有值 = 该层在单层(single/truncated)分析中有结果；
#   - 无结果侧一律 None（CSV 空列读回亦为 None），由下游自行解析。
#   - 单层分析不产出 single_mse（统一 schema 无此键）。
# LAYER_ACC_KEYS = ('entire_cos', 'entire_euc', 'entire_mse',
#                   'single_cos', 'single_euc')
# 内存可选附加键（非 CSV 列）：op_type 算子类型 —— Rknn 侧由 error_analysis
# 自带类型带入；Qnn 侧缺失时 AccuracyGraph 回退 onnx node_info 推断。
# LAYER_ACC_EXTRA = ('op_type',)
# CSV 列：name + LAYER_ACC_KEYS（顺序即骨架序/渲染 x 轴序）
# CSV_ACC_FIELDNAMES = ('name',) + LAYER_ACC_KEYS


def _acc_col(accuracy: dict, key: str) -> list:
    """从统一 dict 按层序取某指标列（缺失 None -> 原样保留，绘图处再转 nan）。"""
    return [row.get(key) if isinstance(row, dict) else None for row in accuracy.values()]


def plot_accuracy_summary(accuracy: dict = None, entire_val_color: str = "blue",
                          save_path=None):
    """统一的逐层精度可视化（欧氏距离柱状图 + 余弦相似度折线图）。

    入参为统一"层精度" dict（schema 见模块顶部 LAYER_ACC_KEYS）：
        {层名: {'entire_cos': float|None, 'entire_euc': float|None,
                'entire_mse': float|None, 'single_cos': float|None,
                'single_euc': float|None}}
    内部按层序(names = accuracy.keys())拆列并绘图：
      - entire 侧任意层有值 -> 画 entire 折线（累积）；
      - single 侧任意层有值 -> 画 single 柱状（单层）；
      - 该列全为 None/空 dict -> 跳过该部分（两个子图始终保留）。
    层名即 x 轴标签；层顺序 = dict 键序（调用方保证为骨架序）。

    Args:
        accuracy: 统一层精度 dict（可为空/None -> 只建空图）。
        entire_val_color: entire 折线颜色（QNN 蓝 / RKNN 橙）。
        save_path: 保存 PNG 路径（None 不保存）。

    Returns:
        save_path（同 plot_accuracy_summary 旧语义）。
    """
    import matplotlib.pyplot as plt
    from matplotlib import axes

    accuracy = accuracy or {}
    names = list(accuracy.keys())
    n = len(names)
    layer_index = np.arange(n)

    def col(key):
        """某指标列 -> np.ndarray；无值(None) -> nan（plot 自动断开/跳过）。"""
        return np.array([v if v is not None else np.nan for v in _acc_col(accuracy, key)],
                       dtype=np.float64)

    single_euc = col('single_euc')
    single_cos = col('single_cos')
    entire_euc = col('entire_euc')
    entire_cos = col('entire_cos')
    mse_vals = col('entire_mse')

    has_single = bool(np.isfinite(single_cos).any())
    has_entire = bool(np.isfinite(entire_cos).any())

    # 两个子图始终创建（即使无数据也保留）
    fig, (ax_euc, ax_cos) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax_euc: axes.Axes
    ax_cos: axes.Axes

    # ---- 上图：欧氏距离（分支：只画有数据的部分）----
    extra_axes: list[axes.Axes] = []
    has_euc_data = False

    if has_single:
        euc_plot = single_euc.copy()
        euc_plot[~np.isfinite(euc_plot)] = np.nan      # 缺值不画柱
        euc_plot[euc_plot == 0] = 1e-10
        ax_euc.set_yscale('linear')
        ax_euc.bar(layer_index, euc_plot, color='skyblue', edgecolor='black',
                   linewidth=0.5, alpha=0.7, label='Euc (single)')
        has_euc_data = True

    if has_entire:
        entire_euc_plot = entire_euc.copy()
        entire_euc_plot[~np.isfinite(entire_euc_plot)] = np.nan
        entire_euc_plot[entire_euc_plot == 0] = 1e-10
        ax_euc.plot(layer_index, entire_euc_plot, color=entire_val_color, marker='.',
                    linestyle='-', linewidth=1.5, markersize=2, label='Euc (entire)')
        has_euc_data = True

    if np.isfinite(mse_vals).any():
        # 右侧纵轴：MSE（量级可能千分之一~个位，与欧氏距离的几十上百分离显示）
        ax_mse = ax_euc.twinx()
        ax_mse.plot(layer_index, mse_vals, color=(0.0, 0.8, 0.6), marker='.',
                    linestyle='-', linewidth=1, markersize=2, label='MSE')
        ax_mse.set_ylabel('MSE', fontsize=12)
        extra_axes.append(ax_mse)
        has_euc_data = True

    ax_euc.set_title('Euclidean Distance', fontsize=14, fontweight='bold')
    ax_euc.set_ylabel('Euclidean Distance', fontsize=12)
    ax_euc.grid(True, which="both", ls="--", alpha=0.5)
    if has_euc_data:
        lines, labels = ax_euc.get_legend_handles_labels()
        for ax in extra_axes:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
        if lines:
            ax_euc.legend(lines, labels, loc='upper left')

    # ---- 下图：余弦相似度（分支：只画有数据的部分）----
    has_cos_data = False

    if has_single:
        cos_sim_plot = single_cos.copy()
        cos_sim_plot[~np.isfinite(cos_sim_plot)] = np.nan
        ax_cos.set_yscale('linear')
        ax_cos.plot(layer_index, cos_sim_plot, color='green', marker='.',
                    linestyle='-', linewidth=1, markersize=2, label='Cosine (single)')
        has_cos_data = True

    if has_entire:
        entire_cos_plot = entire_cos.copy()
        entire_cos_plot[~np.isfinite(entire_cos_plot)] = np.nan
        ax_cos.plot(layer_index, entire_cos_plot, color=entire_val_color, marker='.',
                    linestyle='-', linewidth=1.5, markersize=2, label='Cosine (entire)')
        has_cos_data = True

    ax_cos.set_title('Cosine Similarity', fontsize=14, fontweight='bold')
    ax_cos.set_ylabel('Cosine Similarity', fontsize=12)
    ax_cos.set_xticks(layer_index)
    if names:
        ax_cos.set_xticklabels(names, rotation=-45, ha='left', fontsize=10)
    if has_cos_data:
        ax_cos.axhline(0.99, color='red', linestyle='--', linewidth=1, alpha=0.6,
                       label='Warning Threshold (0.99)')
        ax_cos.legend()
    ax_cos.grid(True, which="both", ls="--", alpha=0.5)

    # 消除 x 轴两端默认的 5% 空白边距（留半个柱宽避免首尾柱被裁切）
    if n > 0:
        ax_cos.set_xlim(-0.5, n - 0.5)

    # 调整布局、保存并显示
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path).resolve()
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()

    return save_path


def build_onnx_tensor_graph(model_path, *, sanitize: bool = False) -> dict:
    """从 ONNX 模型构建张量级 DAG（含可选命名清洗）。

    Args:
        model_path: ONNX 模型路径。
        sanitize: 是否把张量名中的 [/ .]+ 替换为 _。QNN raw 文件名即清洗后的
            张量名，需传 True；RKNN 快照层名基于原始张量名，传 False。

    Returns:
        dict: 包含
            - tensors (dict[str, str]): (清洗后)张量名 -> 原始张量名
            - tensor_set (set[str]): 全部 (清洗后)张量名（输入/输出/节点输出）
            - node_info (dict[str, tuple[str, list[str]]]): 张量 -> (算子类型, 输入张量)
            - pred (dict[str, list[str]]): 张量 -> 产生它的节点输入张量列表
            - succ (dict[str, list[str]]): 张量 -> 消费它的后续张量列表
            - inputs / outputs (list[str]): 模型输入/输出名
            - input_sizes (dict[str, int]): 输入张量 -> 元素总数（仅静态维度）
            - order (list[str]): ONNX 计算图遍历顺序（用于报告/统计排序）
    """
    import onnx

    model = onnx.load(str(model_path))
    g = model.graph

    def _norm(name: str) -> str:
        # 与 qnn-net-run 的 raw 文件名清洗一致（_sanitize_name：任意非字母数字/
        # 下划线字符 -> '_'）。TF 转的 ONNX 用 'name:0' 作输出索引，qnn 会清洗
        # 成 'name_0'，这里必须同样处理 ':'，否则张量匹配失败、层图连不成边。
        return re.sub(r'[^A-Za-z0-9_]', '_', name) if sanitize else name

    tensors: dict[str, str] = {}
    for t in list(g.input) + list(g.output):
        tensors.setdefault(_norm(t.name), t.name)
    for n in g.node:
        for o in n.output:
            if o:
                tensors.setdefault(_norm(o), o)

    node_info: dict[str, tuple[str, list[str]]] = {}
    pred: dict[str, list[str]] = {}
    succ: dict[str, list[str]] = {}
    for n in g.node:
        outs = [_norm(o) for o in n.output if o]
        ins = [_norm(i) for i in n.input if i]
        for out in outs:
            node_info.setdefault(out, (n.op_type, ins))
            pred.setdefault(out, []).extend(ins)
        for i in ins:
            succ.setdefault(i, []).extend(outs)

    def _value_size(vi) -> int | None:
        s = 1
        for d in vi.type.tensor_type.shape.dim:
            v = d.dim_value
            if v and v > 0:
                s *= int(v)
            else:
                return None
        return s

    input_sizes: dict[str, int] = {}
    for t in g.input:
        s = _value_size(t)
        if s:
            input_sizes[_norm(t.name)] = s

    # ONNX 计算图遍历顺序（用于统计图/报告按图顺序排序）
    order: list[str] = []
    for n in g.node:
        for o in n.output:
            if o:
                s = _norm(o)
                if s not in order:
                    order.append(s)
    for t in g.output:
        s = _norm(t.name)
        if s not in order:
            order.append(s)

    return {
        'tensors': tensors,
        'tensor_set': set(tensors),
        'node_info': node_info,
        'pred': pred,
        'succ': succ,
        'inputs': [_norm(t.name) for t in g.input],
        'outputs': [_norm(t.name) for t in g.output],
        'input_sizes': input_sizes,
        'order': order,
    }


def match_tensor(layer_name: str, graph: dict) -> str | None:
    """将快照/raw 层名匹配到 ONNX 张量名（graph['tensors'] 的键）。

    编译器会给张量名追加后缀（RKNN 的 _sw/-rs/_mm 等）或前导下划线
    （QNN sanitize 会把开头的 / 变成 _），因此先做精确匹配（含前导下划线变体），
    失败则取"最长的、是候选名前缀的 ONNX 张量名"。

    Returns:
        str | None: 匹配到的张量名，未匹配返回 None。
    """
    candidates = {layer_name, '_' + layer_name, layer_name.lstrip('_')}
    for c in candidates:
        if c in graph['tensors']:
            return c

    best: str | None = None
    best_len = -1
    for t in graph['tensors']:
        for c in candidates:
            if c.startswith(t) and len(t) > best_len:
                best, best_len = t, len(t)
    return best


def build_layer_graph(rows: list[dict], graph: dict, *, tensor_resolver=None) -> tuple[dict, dict, dict]:
    """构建"层图"：以精度分析行为节点，边由 ONNX 张量依赖推导。

    - 同一 ONNX 张量对应的多个层（如 template / template_int8 / template_conv，
      输入/输出的处理链）按快照顺序串联。
    - 沿 ONNX succ 广度优先，跳过没有快照层的中间张量，连到下一个有快照层的张量。

    Args:
        rows: 精度分析行；若缺 'tensor' 字段，会用 tensor_resolver(layer_name)
            补齐（RKNN 在此时做匹配；QNN 的行已预先填好）。
        graph: build_onnx_tensor_graph() 的返回值。
        tensor_resolver: 可选，layer_name -> graph['tensors'] 键 的映射函数。

    Returns:
        tuple: (children, parents, tensor_layers)
            - children (dict[str, list[str]]): 层 -> 下游层列表
            - parents (dict[str, list[str]]): 层 -> 上游层列表
            - tensor_layers (dict[str, list[str]]): ONNX 张量 -> 对应层列表
    """
    succ = graph['succ']

    tensor_layers: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        t = r.get('tensor')
        if t is None and tensor_resolver is not None:
            t = tensor_resolver(r['layer_name'])
            r['tensor'] = t
        tensor_layers[t].append(r['layer_name'])

    node_order = {r['layer_name']: i for i, r in enumerate(rows)}
    children: dict[str, list[str]] = defaultdict(list)

    def add_edge(a: str | None, b: str | None) -> None:
        if a and b and a != b:
            children[a].append(b)

    # 同一张量的层链（输入/输出处理层）。
    # 注意：跳过 None 张量——否则所有"未匹配到 ONNX 张量"的层会被
    # 错误地串成一条链，产生大量虚假边。
    for tin, lst in tensor_layers.items():
        if tin is None:
            continue
        for i in range(len(lst) - 1):
            add_edge(lst[i], lst[i + 1])

    # ONNX 张量后继边：跳过未映射张量，连到下一个映射张量
    for tin, lst in tensor_layers.items():
        if tin is None:
            continue
        l_in = lst[-1]  # 该张量对应的输出端层（链尾）
        visited: set[str] = set()
        dq = deque(succ.get(tin, []))
        while dq:
            t = dq.popleft()
            if t in visited:
                continue
            visited.add(t)
            if t in tensor_layers:
                add_edge(l_in, tensor_layers[t][0])
                continue  # 遇到映射张量即停止该分支
            dq.extend(succ.get(t, []))

    for k in children:
        children[k] = sorted(set(children[k]), key=lambda x: node_order[x])

    parents: dict[str, list[str]] = defaultdict(list)
    for a, cl in children.items():
        for b in cl:
            parents[b].append(a)

    return children, parents, tensor_layers


def build_terminal_nodes(rows: list[dict], children: dict, parents: dict, graph: dict,
                         *, include_input: bool = False, name_of=None) -> tuple[list[dict], list[dict], list[str], list[str], dict, dict]:
    """在真实层之外追加 Input / Output 示意终端节点并连边。

    每个模型输出追加一个示意 'Output' 终端节点（不替换/不改写原输出层），
    从真正的输出层连出；若该输出张量未被快照，则反向 BFS 找到能到达它的最深
    快照层连出。命名加 '(out)' 后缀避免与原层重名。Output 的精度继承其直接
    上游真实层，使示意终端与真实输出层保持同色。
    include_input=True（QNN，raw 不含输入层）时再追加 'Input' 终端节点（正向
    BFS 连到下游首个快照层），精度记为无损 cos=1.0 / euc=0.0；
    RKNN 快照默认已含输入层，传 False。

    Args:
        rows: 已含 'tensor' 字段的真实层行。
        children / parents: build_layer_graph() 的返回值。
        graph: build_onnx_tensor_graph() 的返回值。
        include_input: 是否同时追加 Input 终端节点。
        name_of: 张量键 -> 显示名 映射；QNN 经 graph['tensors'] 取原始名，
            RKNN 张量键即原始名（name_of=None 时用恒等映射）。

    Returns:
        tuple:
            - input_rows (list[dict]): Input 示意节点行
            - output_rows (list[dict]): Output 示意节点行
            - input_layers (list[str]): Input 节点名
            - output_layers (list[str]): Output 节点名
            - aug_children (dict): 加入终端节点边后的 children
            - aug_parents (dict): 加入终端节点边后的 parents
    """
    if name_of is None:
        name_of = lambda k: k

    # 同一张量可能有多个变体层（如 output0-rs_tp / output0-rs / output0_int8 /
    # output0），处理链的链尾(lst[-1])才是真正的最终输出层，输出终端应从它连出
    tensor_layers: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        t = r.get('tensor')
        if t is not None:
            tensor_layers[t].append(r['layer_name'])
    tensor_to_layer = {t: lst[-1] for t, lst in tensor_layers.items()}
    node_order = {r['layer_name']: i for i, r in enumerate(rows)}
    layer_row = {r['layer_name']: r for r in rows}

    aug_children: dict[str, list[str]] = defaultdict(list)
    aug_parents: dict[str, list[str]] = defaultdict(list)
    for a, cl in children.items():
        aug_children[a] = list(cl)
        for b in cl:
            aug_parents[b].append(a)

    succ = graph['succ']
    pred = graph['pred']

    # ---- Input 终端（可选：QNN raw 不含输入层，需示意）----
    input_rows: list[dict] = []
    input_layers: list[str] = []
    if include_input:
        for inp in graph['inputs']:
            display = name_of(inp)
            input_layers.append(display)
            targets: list[str] = []
            seen: set[str] = set()
            dq = deque(succ.get(inp, []))
            while dq:
                t = dq.popleft()
                if t in seen:
                    continue
                seen.add(t)
                if t in tensor_layers:
                    targets.append(tensor_layers[t][0])
                    continue
                dq.extend(succ.get(t, []))
            targets = sorted(set(targets), key=lambda x: node_order.get(x, 0))
            for tgt in targets:
                aug_children[display].append(tgt)
                aug_parents[tgt].append(display)
            # Input 终端为理想输入源：精度记为无损（cos=1.0 / euc=0.0）
            input_rows.append({
                'layer_name': display, 'op_type': 'Input',
                'entire_cos': 1.0, 'entire_euc': 0.0,
                'single_cos': 1.0, 'single_euc': 0.0,
            })

    # ---- Output 终端 ----
    existing_names = set(node_order) | set(input_layers)
    output_layers: list[str] = []
    output_rows: list[dict] = []
    for out in graph['outputs']:
        onnx_out = name_of(out)
        # 原输出层通常已占用输出张量名，示意节点加后缀避免重名
        node_name = onnx_out if onnx_out not in existing_names else f'{onnx_out} (out)'
        output_layers.append(node_name)
        if out in tensor_to_layer:
            # 有真实输出层：从该层连出到示意终端
            producers = [tensor_to_layer[out]]
        else:
            # 无真实层：反向 BFS 找到能到达该输出的最深快照层
            producers = []
            seen: set[str] = set()
            dq = deque(pred.get(out, []))
            while dq:
                t = dq.popleft()
                if t in seen:
                    continue
                seen.add(t)
                if t in tensor_layers:
                    producers.append(tensor_layers[t][-1])
                    continue
                dq.extend(pred.get(t, []))
            producers = sorted(set(producers), key=lambda x: node_order.get(x, 0))
        for prod in producers:
            aug_children[prod].append(node_name)
            aug_parents[node_name].append(prod)
        # Output 终端精度继承其直接上游（producers 中按快照顺序最靠后的真实层），
        # 使示意终端与真实输出层保持同色；无上游时保持 None。
        if producers:
            src_row = layer_row[producers[-1]]
            out_acc = {k: src_row.get(k)
                       for k in ('entire_cos', 'entire_euc', 'single_cos', 'single_euc')}
        else:
            out_acc = {'entire_cos': None, 'entire_euc': None,
                       'single_cos': None, 'single_euc': None}
        output_rows.append({'layer_name': node_name, 'op_type': 'Output', **out_acc})

    return input_rows, output_rows, input_layers, output_layers, aug_children, aug_parents


class RknnAccuracyDebugger:
    """
    RKNN 精度分析调试器：读取精度分析文件（snapshot/error_analysis.txt）、
    用 matplotlib 可视化、并解析 ONNX 图结构以进行路径追踪。

    Attributes:
        tmp_work_dir (Path): 存放精度分析与中间产物的目录。
        tmp_model_path (Path): convert() 复制到 tmp 目录的模型副本（图结构分析使用）。
        snapshot_dir (Path): RKNN 精度分析快照目录（snapshot/）。
    """

    def __init__(self, tmp_work_dir:str, tmp_model_path:str):
        self.tmp_work_dir = Path(tmp_work_dir).resolve()
        self.tmp_model_path = Path(tmp_model_path).resolve()

        self.snapshot_dir = self.tmp_work_dir / 'snapshot'

        self.file_or_dir_to_clean:list[str] = []
        self.file_or_dir_to_clean.append(self.snapshot_dir)

    def read_error_analysis(self) -> dict:
        """
        读取 RKNN 精度分析结果文件 (snapshot/error_analysis.txt) 并直接组装统一
        "层精度" dict（不再经 list[rows] 中转）。

        {layer_name: {'op_type': str|None,
                      'entire_cos'|'entire_euc'|'entire_mse': float|None,
                      'single_cos'|'single_euc': float|None}}
          - op_type: RKNN 自带算子类型（如 'Conv'、'LeakyRelu'）；
          - 数值键与 LAYER_ACC_KEYS 一致；RKNN 无 mse，entire_mse 恒 None；
          - 少数层（如部分 Concat）无数值，对应字段为 None。

        Raises:
            FileNotFoundError: 当结果文件不存在时。
        """

        error_analysis_path = self.snapshot_dir / 'error_analysis.txt'

        if not error_analysis_path.exists():
            raise FileNotFoundError(
                f"error_analysis.txt not found: {error_analysis_path}\n"
                "请先通过 set_do_accuracy_analysis() 启用精度分析并运行 convert() 生成结果。"
            )

        accuracy: dict[str, dict] = {}
        with open(error_analysis_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                # 跳过注释行、表头行和分隔线
                if line.startswith('#') or line.startswith('layer_name') or line.startswith('---'):
                    continue

                # 数据行格式: [OpType] layer_name   entire_cos | entire_euc   single_cos | single_euc
                if not line.startswith('['):
                    continue
                end_bracket = line.find(']')
                if end_bracket == -1:
                    continue
                op_type = line[1:end_bracket]
                rest = line[end_bracket + 1:].strip()

                # 匹配末尾的 4 个数值: cos | euc   cos | euc
                m = re.match(r'^(.*?)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s*$', rest)
                if m:
                    layer_name = m.group(1).strip()
                    entire_cos = float(m.group(2))
                    entire_euc = float(m.group(3))
                    single_cos = float(m.group(4))
                    single_euc = float(m.group(5))
                else:
                    # 无数值的层（如部分 Concat 层）
                    layer_name = rest
                    entire_cos = entire_euc = single_cos = single_euc = None

                accuracy[layer_name] = {
                    'op_type': op_type,
                    'entire_cos': entire_cos, 'entire_euc': entire_euc, 'entire_mse': None,
                    'single_cos': single_cos, 'single_euc': single_euc,
                }

        print(f"Loaded {len(accuracy)} layers from {error_analysis_path}")
        return accuracy

    def render_combined_report(self, show:bool=True):
        """
        可视化 RKNN 精度分析结果（欧氏距离柱状图 + 余弦相似度折线图）。
        可视化 RKNN 精度分析网络图（Netron 风格 HTML）。

        数据 = read_error_analysis() 组装的统一"层精度" dict（含 RKNN 自带
        op_type）-> AccuracyGraph（onnx 构建/层图/Output 终端在其 __init__ 内
        完成一次）。RKNN 快照已含输入层，include_input=False；层名为原始 ONNX
        张量名，sanitize=False、layer_display 恒等。
        """

        accuracy = self.read_error_analysis()
        if not accuracy:
            print("No data to plot.")
            return 

        save_path = self.tmp_work_dir / 'rknn_accuracy_analysis_summary.png'
        plot_accuracy_summary(accuracy, entire_val_color="orange", save_path=save_path)
        self.file_or_dir_to_clean.append(save_path)


        output_path = self.tmp_work_dir / 'rknn_graph_accuracy_analysis.html'
        viz = AccuracyGraph(
            accuracy,
            self.tmp_model_path,
            output_path,
            title='RKNN Graph Accuracy Analysis',
            sanitize=False,
            include_input=False,
        )
        html_path = viz.render(show=show)
        self.file_or_dir_to_clean.append(html_path)

    def clean(self):
        clean_files_or_dirs(self.file_or_dir_to_clean)




class QnnAccuracyDebugger:
    """基于 qnn-net-run --dlc_path --debug 的双 DLC 精度对比分析器（独立类）。

    直接用两个 DLC 作为材料（FP32 golden + 量化 target）：
      1) 各跑一次 qnn-net-run --dlc_path <dlc> --backend <lib> --input_list <raw列表> --debug，
         得到每层输出（Result_0/*.raw，文件名 = sanitized 张量名）；
      2) 按张量名匹配，手动计算 cosine / 绝对欧氏距离 ||x-y||2 / mse；
      3) 渲染 AccuracyGraph（HTML，cos+euc 双着色）+ plot_accuracy_summary（PNG）+ CSV。

    backend 选择（x86 实测结论）：
      - FP32 DLC -> CPU 后端（libQnnCpu.so；HTP 要求量化模型）
      - 量化 DLC -> HTP 后端（libQnnHtp.so；CPU 后端拒绝量化/16bit 激活中间层）
    """

    def __init__(self, tmp_dir:str, onnx_path:str=None, debugger_picture_list:list=None):
        self.tmp_dir = Path(tmp_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = self.tmp_dir / 'qnn_accuracy_analysis'
        self.onnx_path = Path(onnx_path).resolve()
        self.debugger_picture_list = [Path(p).resolve() for p in (debugger_picture_list or []) if Path(p).exists()]
        self.accuracy_csv_path = self.working_dir / 'qnn_accuracy_analysis_layers.csv'
        
        if sys.platform.startswith('win'):
            # golden(FP32 全层 dump)走 CPU;量化推理/context-binary 只能走 HTP
            # (CPU 后端拒绝量化图且禁止 offline prepare)
            self.backend_lib_golden = "QnnCpu.dll"
            self.backend_lib_target = "QnnCpu.dll"
        else:
            self.backend_lib_golden = "libQnnCpu.so"
            self.backend_lib_target = "libQnnHtp.so"

        self.onnx_info: dict = {}
        self.set_input_order:str = 'nhwc'
        self.exe_qairt_converter:str|None = None
        self.target_dlc_path: str | None = None   # 量化 DLC（画 DLC 视角网络图时反射候选）
        self.file_or_dir_to_clean: list[str] = [str(self.working_dir)]

    # ------------------------------------------------------------------
    # 模型信息 / 通用工具
    # ------------------------------------------------------------------

    def set_model_info(self, onnx_info:dict, set_input_order:str='nhwc'):
        """设置模型输入/输出信息（prepare_input_data 需要 inputs 的名称与形状）。"""
        self.onnx_info = onnx_info or {}
        self.set_input_order = set_input_order

    def prepare_input_data(self, mean_rgb: list = [[0, 0, 0]], std_rgb: list = [[1, 1, 1]]) -> str:
        """根据 debugger_picture_list 与 ONNX 输入信息生成输入 raw 与 input_list。

        返回 input_list 文件路径（qnn-net-run 格式：每行一个样本，多输入用空格分隔）。
        """
        analysis_input_dir = self.working_dir / 'analysis_inputs'
        analysis_input_dir.mkdir(exist_ok=True)

        input_infos = self.onnx_info.get('inputs') or [{'name': 'input0', 'shape': [1, 3, 320, 320]}]
        input_tensor_path_list: list[str] = []

        for i, input_info in enumerate(input_infos):
            mean_value = mean_rgb[i % len(mean_rgb)]
            std_value = std_rgb[i % len(std_rgb)]
            image_path = self.debugger_picture_list[i % len(self.debugger_picture_list)]

            input_name = input_info['name']
            input_shape = input_info['shape']  # nchw
            channels, height, width = input_shape[1], input_shape[2], input_shape[3]

            if image_path.suffix == '.npy':
                image_float = np.load(str(image_path))
            elif image_path.suffix == '.raw':
                image_float = np.fromfile(str(image_path), dtype=np.float32)
            else:
                image = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, (width, height))
                image_float = image_resized.astype(np.float32)
                if channels != 3:
                    image_float = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
                    image_float = np.tile(image_float, (1, 1, channels))[:, :, :channels]
                image_float = np.expand_dims(image_float, axis=0)
                if self.set_input_order == 'nchw':
                    image_float = np.transpose(image_float, (0, 3, 1, 2))

            input_tensor_path = str(analysis_input_dir / f'{input_name}.raw')
            image_float.tofile(input_tensor_path)
            print(f"[QAIRTAccuracyDebugger] input {input_name}: {image_float.shape} -> {input_tensor_path}")
            input_tensor_path_list.append(input_tensor_path)

        input_list = self.working_dir / 'input_list.txt'
        with open(input_list, 'w') as f:
            f.write(' '.join(input_tensor_path_list) + '\n')
        print(f"[QAIRTAccuracyDebugger] input_list: {input_list}")
        return str(input_list)


    def run_qnn_net_run(self, dlc_path:str, backend_lib:str, input_list:str, output_dir:str, output_tensors:str="all") -> int:
        """执行 qnn-net-run --dlc_path <dlc> --backend <lib> --input_list --output_dir [--debug]。"""

        if sys.platform.startswith('win'):
            model_lib = 'QnnModelDlc.dll'
        else:
            model_lib = 'libQnnModelDlc.so'

        cmd = f'qnn-net-run --backend {backend_lib} --model {model_lib}'

        if output_tensors == "all":
            cmd += f' --debug --log_level warn'
        else:
            cmd += f' --set_output_tensors {output_tensors} --log_level warn'

        cmd += f' --dlc_path {dlc_path} --input_list {input_list} --output_dir {output_dir}'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]")

    # ------------------------------------------------------------------
    # 逐层 dump 工具（entire 与 single 共用）
    # ------------------------------------------------------------------

    @staticmethod
    def load_dump_dir(dump_dir) -> dict:
        """读 qnn-net-run --debug 的逐层 dump 目录 -> {sanitized 张量名: fp32 ndarray}。

        raw 文件名 = 张量名；按 fp32 读不出有限值时按 fp16 再读一次
        （hybrid / fp16 dump 的兜底）。
        """
        out = {}
        for p in Path(dump_dir).rglob('*.raw'):
            a32 = np.fromfile(str(p), dtype=np.float32)
            if not np.isfinite(a32).all() and p.stat().st_size % 2 == 0:
                out[p.stem] = np.fromfile(str(p), dtype=np.float16).astype(np.float32)
            else:
                out[p.stem] = a32
        return out

    def ensure_golden_dump(self, golden_dlc_path: str, input_list: str) -> Path:
        """确保 working_dir/golden_output 有 fp32 全图 dump（缺则补跑一次）。"""
        golden_dir = self.working_dir / 'golden_output'
        if not (golden_dir / 'Result_0').exists():
            print(f"[QAIRTAccuracyDebugger] golden dump not found, running golden once ...")
            golden_dir.mkdir(parents=True, exist_ok=True)
            ret = self.run_qnn_net_run(golden_dlc_path, self.backend_lib_golden,
                                       input_list, str(golden_dir))
            if ret != 0:
                raise RuntimeError(f"[QAIRTAccuracyDebugger] golden net-run failed: {ret}")
        return golden_dir

    def accuracy_analysis(self, golden_dlc_path: str, target_dlc_path: str,
                          mean_rgb: list = [[0, 0, 0]], std_rgb: list = [[1, 1, 1]]) -> int:
        """双 DLC 精度对比主入口（entire 累积 + single 单层同图显示）：

        1) entire：golden vs 量化 DLC 一次全量推理，逐层比较（累积误差）；
        2) single：手动逐层（全层、并行）——每层单独生成半量化 DLC
           （其余层 fp16 fallback）再推理，误差仅反映该层自身量化误差；
        3) 统计图 PNG：single 柱状 + entire 折线 + MSE；
        4) 网络图 HTML：节点按 entire_cos 着色，悬停显示 single/entire。

        Args:
            golden_dlc_path: FP32（未量化）DLC 路径。
            target_dlc_path: 量化 DLC 路径（同时作为 single 的层结构反射源）。
        """
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.target_dlc_path = str(target_dlc_path)
        input_list = self.prepare_input_data(mean_rgb, std_rgb)

        # 1) entire（累积）——末尾 save_accuracy_analysis_csv 落盘
        self.entire_accuracy_analysis(golden_dlc_path, target_dlc_path, input_list)

        # 2) single（单层）——失败时退回 entire-only 渲染（不阻断既有流程）。
        #    成功则 save_accuracy_analysis_csv(kind='single') 追加到同一 CSV。
        fallback_entire_only = False
        if sys.platform.startswith('win'):
            fallback_entire_only = True

        if not fallback_entire_only:
            try:
                self.single_accuracy_analysis(golden_dlc_path, target_dlc_path, input_list)
            except Exception as exc:
                fallback_entire_only = True
                print(f"[QAIRTAccuracyDebugger] single-layer analysis skipped ({type(exc).__name__}: {exc}); "
                    f"fallback to entire-only report")


        # 3) 渲染：从精度 CSV 读回（entire/single 已落盘），single 数据可用时输出
        #    entire+single 联合图；fallback(Windows 或 single 失败)时仅 entire 列 -> entire-only 图。
        self.render_combined_report()
        return 0

    def clean(self):
        clean_files_or_dirs(self.file_or_dir_to_clean)


    def entire_accuracy_analysis(self, golden_dlc_path:str, target_dlc_path:str, input_list:str):
        golden_dir = self.working_dir / 'golden_output'
        target_dir = self.working_dir / 'target_output'
        for d in (golden_dir, target_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)


        MultThreadExetutor.run_exetutor(self.run_qnn_net_run, golden_dlc_path, self.backend_lib_golden, input_list, str(golden_dir))
        MultThreadExetutor.run_exetutor(self.run_qnn_net_run, target_dlc_path, self.backend_lib_target, input_list, str(target_dir))

        ret_list = MultThreadExetutor.wait_and_close()
        if ret_list:
            if ret_list[0] != 0:
                raise RuntimeError(f"[QnnAccuracyDebugger] qnn-net-run golden DLC failed with code {ret_list[0]}")
            if ret_list[1] != 0:
                raise RuntimeError(f"[QnnAccuracyDebugger] qnn-net-run target DLC failed with code {ret_list[1]}")
        else:
            raise RuntimeError(f"[QnnAccuracyDebugger] qnn-net-run failed")

        # 读两个 Result_0 目录，按张量名（sanitized 文件名）匹配，算 cos/euc/mse
        gold = self.load_dump_dir(golden_dir)
        targ = self.load_dump_dir(target_dir)
        common = {n for n in gold if n in targ and gold[n].size == targ[n].size}
        # 优先按 ONNX 计算图顺序排序；不在图中的额外张量（如 hybrid 转换节点）排后面
        if self.onnx_path is not None:
            graph = build_onnx_tensor_graph(self.onnx_path, sanitize=True)
            order = [n for n in graph['order'] if n in common]
            order += sorted(common - set(order))
        else:
            order = sorted(common)
        print(f"[QAIRTAccuracyDebugger] matched layers: {len(order)}")

        # 统一 dict：{层名: {entire_cos/euc/mse 有值, single 侧 None}}
        accuracy = {}
        for n in order:
            a, b = gold[n], targ[n]
            accuracy[n] = {
                'entire_cos': float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                'entire_euc': float(np.linalg.norm(a - b)),
                'entire_mse': float(np.mean((a - b) ** 2)),
                'single_cos': None, 'single_euc': None,
            }

        self.save_accuracy_analysis_csv(accuracy, kind='entire')

    # ------------------------------------------------------------------
    # single：截断单算子实现（详见 QnnTruncatedAccuracyAnalysis）
    # ------------------------------------------------------------------

    def single_accuracy_analysis(self, golden_dlc_path: str, target_dlc_path: str,
                                 input_list: str, only_tensor_prefix=None):
        """手动逐层（layerwise == single）精度分析（默认全层、并行）。

        误差语义 = 官方 layerwise：整图其余部分浮点，仅该层量化 -> 误差仅反映
        该层自身的量化误差（非累积），对应报告的 single_euc / single_cos。

        实现 = QnnTruncatedAccuracyAnalysis（截断单算子管线；本类组合持有该
        分析器并复用 golden/候选/encodings/converter/ctx/net-run 工具）。

        整体失败（抛异常）时不自动退回整图半量化实现；accuracy_analysis 捕获
        后跳过 single，只输出 entire 报告。
        （旧整图半量化实现 QnnSingleAccuracyAnalysisFull 已归档到
        legacy/qnn_single_accuracy_analysis_full.py，不再参与现役流程。）

        Args:
            golden_dlc_path: FP32（未量化）DLC 路径（golden dump 缺则自动补跑）。
            target_dlc_path: 量化 DLC 路径（反射层结构 + 自动导出 v2 encodings）。
            input_list: qnn-net-run 输入列表。
            only_tensor_prefix: 只分析张量名以此前缀开头的层（str 或 tuple）。

        本方法不返回值：内部调 QnnTruncatedAccuracyAnalysis.run 拿到统一"层精度"
        dict（single 侧有值），save_accuracy_analysis_csv(kind='single') 落盘。
        """
        analyzer = QnnTruncatedAccuracyAnalysis(self, self.onnx_path)
        accuracy = analyzer.run(golden_dlc_path, target_dlc_path, input_list, only_tensor_prefix)
        self.save_accuracy_analysis_csv(accuracy, kind='single')

    # ------------------------------------------------------------------
    # 精度结果 CSV（working_dir/qnn_accuracy_analysis_layers.csv）
    # ------------------------------------------------------------------

    def save_accuracy_analysis_csv(self, accuracy: dict, kind: str = 'entire') -> str:
        """把一轮精度结果（统一 dict）merge 写入 working_dir/qnn_accuracy_analysis_layers.csv。

        统一 dict 形态见模块顶部 LAYER_ACC_KEYS 注释：
            {层名: {'entire_cos': float|None, 'entire_euc': float|None,
                    'entire_mse': float|None, 'single_cos': float|None,
                    'single_euc': float|None}}
        kind 决定本次写哪一侧、如何定序：
          - kind='entire'：以 accuracy 键序（ONNX 拓扑序全集）重建 CSV 骨架，
            写 entire 三列；single 列清空（本次 fresh）；
          - kind='single' ：在现有骨架上按层名更新/追加 single 两列，不重排
            文件（单独调试 single 时文件可能只有 single 行）。
        数值列 float、缺值 None；entire 侧额外产出 entire_mse（单层无 mse）。

        Args:
            accuracy: 统一层精度 dict（至少含 kind 对应侧的键）。
            kind: 'entire' | 'single'。

        Returns:
            CSV 路径。
        """
        if kind not in ('entire', 'single'):
            raise ValueError(f"kind must be 'entire' or 'single', got {kind!r}")
        csv_path = self.accuracy_csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        def fnum(v):
            return '' if v is None else repr(float(v))

        # 读取现有骨架（统一 dict {name: row}，保持顺序）
        old = self.load_accuracy_analysis_csv(csv_path)

        # CSV 行统一形态：{'name':..., 'entire_cos':.., 'entire_euc':.., 'entire_mse':..,
        #                   'single_cos':.., 'single_euc':..}（字符串）
        def to_csv_row(name, row):
            return {
                'name': name,
                'entire_cos': fnum(row.get('entire_cos')),
                'entire_euc': fnum(row.get('entire_euc')),
                'entire_mse': fnum(row.get('entire_mse')),
                'single_cos': fnum(row.get('single_cos')),
                'single_euc': fnum(row.get('single_euc')),
            }

        if kind == 'entire':
            # 重建：entire 全集顺序 = 骨架；single 侧清空（本次 fresh）
            rows = [to_csv_row(name, row) for name, row in accuracy.items()]
        else:
            # 更新/追加 single 列，保留既有行序（entire 骨架不动）
            rows = [to_csv_row(name, row) for name, row in old.items()]
            seen = {r['name'] for r in rows}
            for name, row in accuracy.items():
                sc, se = row.get('single_cos'), row.get('single_euc')
                if name in seen:
                    for r in rows:
                        if r['name'] == name:
                            r['single_cos'] = fnum(sc)
                            r['single_euc'] = fnum(se)
                            break
                else:
                    new_row = to_csv_row(name, {})
                    new_row['single_cos'] = fnum(sc)
                    new_row['single_euc'] = fnum(se)
                    rows.append(new_row)
                    seen.add(name)

        fieldnames = ('name', 'entire_cos', 'entire_euc', 'entire_mse', 'single_cos', 'single_euc')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[QAIRTAccuracyDebugger] accuracy csv ({kind}, {len(accuracy)} layers) -> {csv_path}")
        return str(csv_path)

    def load_accuracy_analysis_csv(self, accuracy_csv_path: str) -> dict:
        """读回精度 CSV -> 统一层精度 dict（有序，键序 = 文件行序）。

        {层名: {'entire_cos': float|None, ..., 'single_euc': float|None}}。
        文件不存在返回空 dict {}（而非 []）。旧列名（mse_e/single_mse）自动忽略。
        """
        csv_path = Path(accuracy_csv_path)
        if not csv_path.exists():
            return {}
        accuracy = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            for line in csv.DictReader(f):
                name = line.get('name', '')
                if not name:
                    continue
                row = {}
                for k in ('entire_cos', 'entire_euc', 'entire_mse', 'single_cos', 'single_euc'):
                    v = (line.get(k) or '').strip()
                    row[k] = float(v) if v else None
                accuracy[name] = row
        return accuracy

    # ------------------------------------------------------------------
    # 渲染：AccuracyGraph（HTML）与结果组织
    # ------------------------------------------------------------------

    def render_combined_report(self) -> None:
        """从精度 CSV（working_dir/qnn_accuracy_analysis_layers.csv）读回统一 dict 并渲染。

        entire / single 分析在各自末尾经 save_accuracy_analysis_csv 落盘；此处不再
        接收参数，直接读 CSV（统一 dict）后渲染（调试/重画时无需重跑 net-run 重算精度）。

        - CSV 含 entire 侧：以文件行序（entire 全集，ONNX 拓扑序）为 x 轴/图序；
        - CSV 含 single 侧：叠画 single 柱状 / 网络图 tooltip 显示单层精度；
        - 仅 single 无 entire：只画 single 统计图（entire 为空，网络图无法着色，跳过）。
        """
        accuracy = self.load_accuracy_analysis_csv(self.accuracy_csv_path)
        if not accuracy:
            raise RuntimeError(
                "[QAIRTAccuracyDebugger] accuracy CSV 不存在/为空，无法渲染；"
                "请先运行 accuracy_analysis()/entire_accuracy_analysis() 生成 "
                f"{self.accuracy_csv_path}")

        has_entire = any(r.get('entire_cos') is not None for r in accuracy.values())
        has_single = any(r.get('single_cos') is not None for r in accuracy.values())

        plt_path = Path(self.tmp_dir / 'qnn_accuracy_analysis_summary.png')
        plot_accuracy_summary(accuracy, entire_val_color="blue", save_path=plt_path)
        self.file_or_dir_to_clean.append(str(plt_path))

        # 网络图：需要 entire 作节点着色；仅 single（无 entire）时跳过
        if self.onnx_path is not None and has_entire:
            self.draw_network_analysis(accuracy)

    def draw_network_analysis(self, accuracy: dict, show: bool = True,
                              dlc_candidates: list[dict] | None = None) -> dict:
        """基于统一"层精度" dict 渲染 AccuracyGraph（Netron 风格 HTML）。

        onnx 图构建 / 层图 / Input·Output 终端全部收敛到 AccuracyGraph.__init__
        （_parse_accuracy_dict）。QNN 视角默认用 DLC：节点全集 = 量化 DLC 候选层
        （精度缺失层照画，None）、op_type = DLC 反射类型、连线 = DLC io_tensors；
        Input/Output 终端取 DLC IR 图边界（graph_io_tensors，排除权重/常量）；
        dlc_candidates 未传时自动从 self.target_dlc_path 反射（无该路径则退回
        ONNX 视角）。

        Args:
            accuracy: 统一层精度 dict（schema 见 LAYER_ACC_KEYS 注释）。
            show: 渲染后是否自动打开浏览器。
            dlc_candidates: 可选；量化 DLC layer_candidates() 产物。None 时若
                self.target_dlc_path 存在则自动反射，否则 ONNX 视角。

        Returns:
            data: AccuracyGraph 内部数据结构（rows/inputs/outputs）。
        """
        if self.onnx_path is None:
            raise ValueError("[QAIRTAccuracyDebugger] draw_network_analysis needs onnx_path")
        output_path = self.tmp_dir / 'qnn_graph_accuracy_analysis.html'

        dlc_io = None
        if dlc_candidates is None and self.target_dlc_path:
            try:
                exporter = DlcV2EncodingExporter(self.target_dlc_path)
                dlc_candidates = exporter.layer_candidates()
                dlc_io = exporter.graph_io_tensors()   # 图边界（排除权重/常量）
            except Exception as exc:
                print(f"[QAIRTAccuracyDebugger] DLC candidate reflect failed, fallback onnx view: {exc}")
                dlc_candidates = None
                dlc_io = None

        viz = AccuracyGraph(
            accuracy,
            self.onnx_path,
            output_path,
            title='QNN Graph Accuracy Analysis',
            sanitize=True,
            include_input=True,
            layer_display=lambda g, k: g['tensors'].get(k, k),
            dlc_candidates=dlc_candidates,
            dlc_io=dlc_io,
        )
        html_path = viz.render(show=show)
        self.file_or_dir_to_clean.append(html_path)
        return viz.data



def ensure_sdk_pythonpath() -> None:
    """主进程内使用 qti 前，把 SDK 的 python 目录并入 sys.path。

    os.environ 里的 PYTHONPATH 只在解释器启动时读取；run_env_script / setup_sdk
    运行时注入的环境不会自动更新已运行解释器的 sys.path，此处显式并入，
    使 encodings/layers/tinydlc 等读取可在主进程直接完成（不再递归子进程）。
    """
    for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)

_CONV_SCRIPT_CACHE = None

def _qairt_converter_script() -> str:
    """qairt-converter 脚本绝对路径（去掉 get_tool 的引号包装）。"""
    global _CONV_SCRIPT_CACHE
    if _CONV_SCRIPT_CACHE is None:
        from onnx_to_qnn import QAIRTScript
        _CONV_SCRIPT_CACHE = QAIRTScript.get_tool('qairt-converter').strip().strip('"')
    return _CONV_SCRIPT_CACHE


def _conv_argv(cut_onnx: str, out_dlc: str, ov_json: str, layout: str) -> list:
    import shlex
    args = ['--input_network', cut_onnx, '--output_path', out_dlc,
            '--quantization_overrides', ov_json]
    args += shlex.split(layout)          # 去引号，与 shell 语义一致
    args += ['--onnx_skip_simplification']
    return args


def _trunc_conv_one(job: tuple) -> tuple:
    """进程池 worker：转换一个 cut 图 -> 量化 DLC，返回 (rc, tail)。

    job = (cut_onnx, ov_json, layout, out_dlc)（均为 str 路径）。
    引擎复用：qairt-converter 用 runpy 在本进程内执行，重模块只 import 一次，
    同一 worker 连续处理多层时后续层直接复用（并行的主要收益来源）。
    输出 fd 级静默：converter 的 C 层直接写 fd1/fd2，只重定向 sys.stdout/stderr
    会漏；失败时回传尾部便于诊断。
    """
    cut, ov, layout, out_dlc = job
    ensure_sdk_pythonpath()
    script = _qairt_converter_script()
    argv = [script] + _conv_argv(cut, out_dlc, ov, layout)
    old_argv = sys.argv
    sys.argv = argv
    txt = ''
    try:
        import runpy
        import tempfile
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8',
                                    errors='replace') as tf:
            saved_out, saved_err = os.dup(1), os.dup(2)
            os.dup2(tf.fileno(), 1)
            os.dup2(tf.fileno(), 2)
            try:
                runpy.run_path(script, run_name='__main__')
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
            tf.flush()
            tf.seek(0)
            txt = tf.read()
        return 0, ''
    except SystemExit as e:
        return int(e.code or 0), txt[-1500:]
    except Exception as exc:
        return -1, f'{type(exc).__name__}: {exc}\n' + txt[-1500:]
    finally:
        sys.argv = old_argv


# ==========================================================================
# tiny DLC meta 读取（qti IR 图直读）
# --------------------------------------------------------------------------
# 生产路径：QnnTruncatedAccuracyAnalysis._stage_meta -> query_tiny_dlc_batch
# -> query_tiny_dlc；CLI：python accuracy_debugger.py tinydlc / tinydlc-batch。
# 每个文件新建 IrDlcReader（复用同一 reader 重复 open 会返回旧图导致串数据）。
# ==========================================================================

# QNN_DATATYPE_* -> 'uint8'/'int16'/'float32' 等（编码导出与 tiny meta 共用一份）
_DTYPE_PATTERNS = (
    (re.compile(r'UFIXED_POINT_(\d+)'), 'uint'),
    (re.compile(r'SFIXED_POINT_(\d+)'), 'int'),
    (re.compile(r'FLOAT_(\d+)'), 'float'),
)


def _qnn_dtype_name(data_type) -> str:
    """QNN_DATATYPE_* -> 'uint8' / 'int16' / 'float32' 等字符串。"""
    name = getattr(data_type, 'name', None) or str(data_type)
    for pat, prefix in _DTYPE_PATTERNS:
        m = pat.search(name)
        if m:
            return prefix + m.group(1)
    return name


def _read_tiny_dlc_meta(dlc_path: str) -> dict:
    """tiny DLC -> {'inputs': [...], 'outputs': [...], 'tensors': {name: {...}}}。

    tensors 项含 dtype/dims；量化张量附 scale（axis 量化再带 axis）与 zp
    （QNN 约定 zp 为负，offset == 0 时省略）。需 SDK 环境已注入。
    """
    ensure_sdk_pythonpath()
    from qti.aisw.dlc_utils import modeltools as _mt
    reader = _mt.IrDlcReader()
    reader.open(str(dlc_path))
    g = reader.get_ir_graph()

    def tensors_of(fn):
        try:
            return [{'name': t.name(), 'dims': list(t.dims()),
                     'dtype': _qnn_dtype_name(t.data_type())}
                    for t in getattr(g, fn)()]
        except Exception:
            return []

    inputs = tensors_of('get_input_tensors_to_graph') or tensors_of('get_input_tensors')
    outputs = tensors_of('get_output_tensors_of_graph') or tensors_of('get_output_tensors')
    tensors = {}
    for name, t in g.get_tensor_map().items():
        entry = {'dtype': _qnn_dtype_name(t.data_type()), 'dims': list(t.dims())}
        try:
            enc = t.get_encoding()
            if enc is not None:
                en = (getattr(getattr(enc, 'type', None), 'name', None)
                      or str(getattr(enc, 'type', '')))
                if 'AXIS_SCALE_OFFSET' in en:
                    es = enc.axisEncInfo.encInfos
                    entry['scale'] = [x.scale for x in es]
                    entry['axis'] = enc.axisEncInfo.axis
                    if not all(x.offset == 0 for x in es):
                        entry['zp'] = [x.offset for x in es]
                elif 'SCALE_OFFSET' in en:
                    info = enc.encInfo
                    entry['scale'] = info.scale
                    if info.offset != 0:
                        entry['zp'] = info.offset          # QNN 约定（负）
        except Exception:
            pass
        tensors[name] = entry
    return {'inputs': inputs, 'outputs': outputs, 'tensors': tensors}


def _write_tiny_dlc_meta(dlc_path: str) -> str:
    """tiny DLC meta -> 同目录 <stem>.tiny_meta.json（返回输出路径）。"""
    out = Path(dlc_path).with_suffix('.tiny_meta.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(_read_tiny_dlc_meta(str(dlc_path)), f, indent=1)
    return str(out)


# ==========================================================================
# 量化 DLC 反射器（DlcV2EncodingExporter）：encodings + 层候选 一站式读取。
# --------------------------------------------------------------------------
# QAIRT SDK(qti)惰性导入：仅实例化时才需要 SDK python 路径可用（由
# onnx_to_qnn.run_env_script 注入环境，ensure_sdk_pythonpath 把 SDK python
# 目录并入 sys.path）。单层分析类（QnnTruncatedAccuracyAnalysis）主进程
# 直接实例化本类完成 encodings /
# layer-candidates 读取（不再递归子进程）；本类也可被直接实例化
# （前提：当前解释器已具备 SDK 的 sys.path / LD_LIBRARY_PATH）。
# ==========================================================================

class DlcV2EncodingExporter:
    """从量化 DLC 提取张量量化编码（AIMET 2.0.0）并反射层候选，一次开文件。

    提供两类数据（同一 reader / 同一 IR 图）：
      - encodings：可量化张量的量化编码，导出为 AIMET 2.0.0 json
        （extract / to_dict / dump）；
      - layer candidates：每个有输出张量的 op -> {tensor, op_type, io_tensors}
        （layer_candidates，供单层 overrides 筛选）。

    qti 模块在首次实例化时惰性导入并缓存在类属性（进程内一次）；编码类型
    常量与 dtype 名称正则亦在类上预编译/缓存，避免逐张量重复属性查找。
    """

    # qti 惰性导入缓存（类级，首次 _ensure_sdk 时填充）
    _modeltools = None
    _ir_graph = None

    def __init__(self, dlc_path, reader=None):
        """打开量化 DLC 并读取 IR 图。

        Args:
            dlc_path: 量化 DLC 文件路径。
            reader: 可选的 DLC 读取器实例（用于测试注入）；默认创建
                    modeltools.IrDlcReader。
        """
        modeltools, _ = self._ensure_sdk()
        self.dlc_path = str(dlc_path)          # qti pybind 只收 str，不接受 Path
        self._reader = reader or modeltools.IrDlcReader()
        self._reader.open(self.dlc_path)
        self._ir_graph = self._reader.get_ir_graph()
        if self._ir_graph is None:
            raise ValueError(f"Failed to read IR graph from DLC: {dlc_path}")

    # ------------------------------------------------------------------
    # SDK 惰性导入与编码常量缓存
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_sdk(cls):
        """惰性导入 qti 模块（首次调用时执行一次），并缓存编码类型常量。"""
        if cls._ir_graph is None:
            ensure_sdk_pythonpath()
            from qti.aisw.dlc_utils import modeltools as _mt
            from qti.aisw.converters.common import ir_graph as _ig
            cls._modeltools, cls._ir_graph = _mt, _ig
            # encoding.type 判定常量（QNN_QUANTIZATION_ENCODING_*）缓存到类上，
            # 避免逐张量经模块属性查找
            cls._ENC_SCALE_OFFSET = _ig.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            cls._ENC_BW_SCALE_OFFSET = _ig.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET
            cls._ENC_AXIS_SCALE_OFFSET = _ig.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
            cls._ENC_BW_AXIS_SCALE_OFFSET = _ig.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
        return cls._modeltools, cls._ir_graph

    @property
    def ir_graph(self):
        """底层 IR 图对象（libPyIrGraph）。"""
        return self._ir_graph

    # ------------------------------------------------------------------
    # encodings 导出（AIMET 2.0.0）
    # ------------------------------------------------------------------

    @classmethod
    def _tensor_encoding_v2(cls, ir_tensor) -> dict:
        """单个 IR 张量 -> AIMET 2.0.0 条目（对齐 TensorEncoding.V2）。"""
        cls._ensure_sdk()
        out = {'output_dtype': _qnn_dtype_name(ir_tensor.data_type()),
               'name': ir_tensor.name()}
        enc = ir_tensor.get_encoding()
        et = enc.type
        if et in (cls._ENC_SCALE_OFFSET, cls._ENC_BW_SCALE_OFFSET):
            e = enc.encInfo
            out['y_scale'] = e.scale
            if e.offset != 0:
                out['y_zero_point'] = e.offset
        elif et in (cls._ENC_AXIS_SCALE_OFFSET, cls._ENC_BW_AXIS_SCALE_OFFSET):
            es = enc.axisEncInfo.encInfos
            out['y_scale'] = [x.scale for x in es]
            out['axis'] = enc.axisEncInfo.axis
            if not all(x.offset == 0 for x in es):
                out['y_zero_point'] = [x.offset for x in es]
        else:
            raise ValueError(f"unsupported encoding type: {et}")
        return out

    def extract(self) -> list:
        """遍历图中所有可量化张量，返回 AIMET 2.0.0 条目列表。"""
        encs = []
        for _name, ir_t in self.ir_graph.get_tensor_map().items():
            if ir_t.is_quantizable():
                encs.append(self._tensor_encoding_v2(ir_t))
        return encs

    def to_dict(self) -> dict:
        """返回 {"version": "2.0.0", "encodings": [...]} 字典。"""
        return {'version': '2.0.0', 'encodings': self.extract()}

    def dump(self, out_json: str) -> str:
        """写出 encodings json（与官方 ModelEncoding.dump 一致：indent=4, ensure_ascii=False）。

        Returns:
            out_json（与旧 export_dlc_encodings 返回值一致，便于链式调用）。
        """
        data = self.to_dict()
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"exported {len(data['encodings'])} encodings -> {out_json}")
        return out_json

    # ------------------------------------------------------------------
    # 层候选反射（每算子 io 张量；供单层 overrides）
    # ------------------------------------------------------------------

    def layer_candidates(self) -> list[dict]:
        """遍历图中每个有输出张量的 op，产出层候选列表。

        每个候选 = {"tensor": <该层输出张量名>, "op_type": <op类型>,
                    "io_tensors": [op 输入+输出张量名...]}。
        与官方 subgraph_snooper 的 overrides 范围一致：
        Conv2d -> input+weight+bias+output（4个）；Prelu -> input+coeff+output（3个）；
        Concat -> 各输入激活+output（4个）。
        """
        candidates = []
        for op in self._ir_graph.get_ops():
            outs = [t.name() for t in op.outputs()]
            if not outs:
                continue
            candidates.append({
                'tensor': outs[0],
                'op_type': str(op.type.name if hasattr(op.type, 'name') else op.type),
                'io_tensors': [t.name() for t in op.inputs()] + outs,
            })
        return candidates

    def graph_io_tensors(self) -> dict:
        """IR 图级边界张量名：{"inputs": [...], "outputs": [...]}。

        图边界 = IR 图声明的输入/输出张量（权重/偏置等静态张量不在其中）。
        供精度图 Input/Output 终端使用：不要用"未被任何 op 产出的候选输入"
        反推图输入——TF 导出的权重名为 `.../kernel:0`、converter hoist 常量
        （`*_hoist_0231`）都不含 weight/bias 等字样，会被误判成图输入。
        """
        return {
            'inputs': [t.name() for t in self._ir_graph.get_input_tensors_to_graph()],
            'outputs': [t.name() for t in self._ir_graph.get_output_tensors_of_graph()],
        }


class QnnTruncatedAccuracyAnalysis:
    """截断单算子（truncated layerwise）精度分析：每层从整图切出单算子小图执行。

    语义 = 官方 layerwise：整图其余部分浮点、仅该层量化 -> 误差只反映该层
    自身的量化误差（single_euc / single_cos，与 entire 累积相对）。

    与 QnnAccuracyDebugger 的关系：组合（composition）——本类持有 dbg 实例，
    复用其 golden net-run / backend / 工作目录 / set_input_order；SDK 工具
    （ctx generator / net-run-bin / encodings / layer candidates / override
    json / 进度条线程池）自带副本，不依赖 dbg 转发。

    每层流程（run() 只做阶段化编排，各阶段私有方法按调用顺序排在下方）：
      1) golden dump（复用 fp32 全图 --debug 结果，缺则补跑一次，不截断 golden）;
      2) 反射量化 DLC 层候选 + 导出 v2 encodings + 模型输入名/raw 映射 + dims;
      3) 逐层准备：derive_activation_inputs -> extend_boundary_inputs(N_BACK)
         -> extend_boundary_inputs_safe（边界安全化，见类常量注释）
         -> build_cut_onnx（保留目标层前 N_BACK 层浮点上界）+ 单层 overrides；
         安全化后仍不安全的层直接跳过（宁缺勿假）;
      4) 阶段A converter（进程池：每进程内复用引擎连续转多片，fd 级静默）;
      5) 阶段B tiny DLC meta（qti 主进程直读，每文件新建 reader）;
      6) 阶段C ctx binary（混合精度 DLC 必须离线编译）;
      7) 阶段D 写 fp32 feed + 阶段E net-run --retrieve_context（并行）;
      8) best-match 比对 vs golden（两种布局取最大 cos，量化输出按编码反量化）。

    golden 侧仍是一次 fp32 全图 --debug dump（不截断），供喂数与比对共用。
    数值与整图半量化 single 一致（RetinaFace 实测 116/118 层，cos 差 <= 1e-5；
    余 2 层结构上不可截断：DLC 自造张量 / 以该张量为输入的 Softmax）。
    """

    # 各阶段并发 worker 数。converter 走进程池（每进程内连转多片，引擎复用），
    # 取 4；ctx/feed/net_run 线程池取 8（12 核 x86 实测饱和值，继续加大无增益/变慢）。
    _WORKERS = dict(converter=4, ctx=8, feed=8, net_run=8)

    # 截断边界向上保留的浮点算子层数(类级常量;实验对比可临时覆写,如
    # cls.N_BACK = 3)。越界/遇分叉时实际保留层数可能 < N_BACK(见
    # extend_boundary_inputs 的提前停止)。边界安全化会按需再向上扩，故
    # N_BACK=1/2/3 的成功层数一致（实测均 116/118）。
    N_BACK = 2

    # 边界安全化（boundary safety）：边界张量若被"非目标"的 shape 算子直接
    # 消费，converter 会把它当规范 NCHW 再按 NHWC 约定置换一次 dims，后续
    # Reshape/Transpose 的展平顺序与整图不一致 -> 数值静默错位。
    # 实测（RetinaFace 118 层）：N_BACK=1 时 output0/output2 cos 0.1118/0.0297
    # （流水线"成功"但值错），N_BACK=2 时同两层 feed 失败；安全扩展后 1/2/3
    # 三档均 116/118 且两层 cos 0.999915/0.999906（与 N_BACK=3 基线一致）。
    _BOUNDARY_SHAPE_OPS = ('Reshape', 'Transpose', 'Flatten', 'Squeeze',
                           'Unsqueeze', 'Expand', 'Slice', 'Gather', 'Split')
    MAX_BOUNDARY_EXTEND = 6   # 安全扩展上限（防极端图无限向上扩）

    def __init__(self, dbg: QnnAccuracyDebugger, onnx_path):
        """组合持有 QnnAccuracyDebugger；只存属性，不做任何初始化工作。"""
        self.dbg = dbg
        self.onnx_path = Path(onnx_path) if onnx_path else None

    # ------------------------------------------------------------------
    # 入口：run() —— 阶段化编排（每个阶段委托一个私有方法）
    # ------------------------------------------------------------------

    def run(self, golden_dlc_path: str, target_dlc_path: str, input_list: str,
            only_tensor_prefix=None) -> dict:
        """执行截断单算子逐层分析：整图其余部分浮点、仅目标层量化。

        整体失败（抛异常）不自动降级；由调用方（accuracy_analysis）捕获后
        只输出 entire。

        Args:
            golden_dlc_path: FP32（未量化）DLC 路径（golden dump 缺则自动补跑）。
            target_dlc_path: 量化 DLC 路径（反射层结构 + 自动导出 v2 encodings）。
            input_list: qnn-net-run 输入列表。
            only_tensor_prefix: 只分析张量名以此前缀开头的层（str 或 tuple）。

        Returns:
            统一"层精度" dict：{层名(sanitized): {'single_cos':..,'single_euc':..,
            'entire_cos':None,'entire_euc':None,'entire_mse':None}}（single 侧有值，
            entire 侧占位 None；单层分析不产出 single_mse）。
        """
        dbg = self.dbg
        if self.onnx_path is None:
            raise RuntimeError("QnnTruncatedAccuracyAnalysis needs onnx_path "
                               "(QnnAccuracyDebugger(..., onnx_path=...))")

        # 0) 运行上下文：输入布局（layout 声明与喂数重排都按它来）
        self._input_order = getattr(dbg, 'set_input_order', 'nhwc')

        # 1) golden dump（复用 dbg.working_dir/golden_output，缺则补跑一次）
        golden_dir = dbg.ensure_golden_dump(golden_dlc_path, input_list)
        golden = dbg.load_dump_dir(golden_dir)
        if not golden:
            raise RuntimeError(f"[QAIRTAccuracyDebugger] golden dir empty: {golden_dir}")
        print(f"[QAIRTAccuracyDebugger] golden layers: {len(golden)}")

        # 2) 层候选 + v2 encodings + 模型输入名/raw 映射 + 全图 NCHW dims
        layers, v2_encodings_path, dims_nchw = self._collect_run_context(
            target_dlc_path, input_list, golden, only_tensor_prefix)

        # 3) 逐层准备：cut ONNX（前 2 层浮点上界）+ 单层 overrides
        infos = self._prepare_all_layers(layers, v2_encodings_path, dims_nchw)

        # 4) 阶段A converter（进程池）    5) 阶段B tiny meta（主进程直读）
        conv_ok = self._stage_converter(infos)
        meta_ok = self._stage_meta(conv_ok)

        # 6) 阶段C ctx binary    7) 阶段D feed + 阶段E net-run
        ctx_ok = self._stage_context_binary(meta_ok)
        net_ok = self._stage_feed_and_netrun(ctx_ok, golden, dims_nchw)

        # 8) best-match 比对 vs golden
        return self._collect_results(net_ok, golden, dims_nchw, total=len(layers))

    # ------------------------------------------------------------------
    # 阶段私有方法（按 run() 调用顺序排列）
    # ------------------------------------------------------------------

    def _collect_run_context(self, target_dlc_path: str, input_list: str,
                             golden: dict, only_tensor_prefix) -> tuple:
        """反射层候选（过滤）+ 导出 v2 encodings + 模型输入映射 + 全图 dims。

        模型输入名取 ONNX graph.input 顺序（与 prepare_input_data 写 raw 的
        顺序一致）；input_list 第一个样本的 raw 路径按序对应每个输入，
        多输入模型每个输入各一条，不写死 input0。
        """
        layers = self.extract_layer_candidates(target_dlc_path)
        if only_tensor_prefix:
            layers = [c for c in layers if c['tensor'].startswith(only_tensor_prefix)]
        layers = [c for c in layers if self._sanitize_name(c['tensor']) in golden]
        print(f"[QAIRTAccuracyDebugger] candidate layers to analyze: {len(layers)}")

        v2_encodings_path = self.extract_dlc_encoding(target_dlc_path)

        full_m = onnx.load(str(self.onnx_path))
        self._model_input_names = [i.name for i in full_m.graph.input]
        with open(input_list, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        tokens = first_line.split()
        if len(tokens) != len(self._model_input_names):
            raise RuntimeError(
                f"[QAIRTAccuracyDebugger] input_list raw count ({len(tokens)}) != "
                f"model inputs ({len(self._model_input_names)}): {self._model_input_names}")
        self._input_raw_by_name = dict(zip(self._model_input_names, tokens))
        dims_nchw = self.onnx_graph_dims_nchw(str(self.onnx_path))
        return layers, v2_encodings_path, dims_nchw

    def _prepare_all_layers(self, layers: list, v2_encodings_path: str,
                            dims_nchw: dict) -> list:
        """逐层准备：样本目录纯序号（001..），张量名只出现在输出 raw 文件名。

        绝对路径嵌张量名会超 ~260 字符（深层长名层 qnn-net-run 写文件失败
        rc17），目录只放序号（与手动 layerwise 的 out/<NNN> 同一先例）。
        不可截断的层（无激活输入 / 输出不在 ONNX 图中）在 _prepare_layer
        内跳过；全部失败时直接抛错（不让后续阶段空转）。
        """
        dbg = self.dbg
        samples_dir = dbg.working_dir / 'truncated'   # 001/002/... 直接放这里
        samples_dir.mkdir(parents=True, exist_ok=True)

        infos = []
        for i, layer in enumerate(layers):
            tag = f"{i + 1:03d}"
            sdir = samples_dir / tag
            sdir.mkdir(parents=True, exist_ok=True)
            info = self._trunc_prepare_layer(sdir, layer, v2_encodings_path, dims_nchw)
            if info is not None:
                infos.append(info)
        print(f"[QAIRTAccuracyDebugger] truncated prepared layers: {len(infos)}")
        if not infos:
            raise RuntimeError("[QAIRTAccuracyDebugger] truncated: no layer prepared")
        return infos

    def _stage_converter(self, infos: list) -> list:
        """阶段A：converter（进程池）——每层一个任务，按提交顺序回收结果。

        每层一个任务（不再静态分片）：worker 进程内 qairt-converter 只 import
        一次、后续层复用引擎，收益与分片相同；动态派发还避免巨型层把某个分片
        拖长（MSI 实测有 36.7MB 的 cut 单层 ~4s）。

        返回码为 0 的 info 列表（阶段失败不中断，交由下游过滤/汇报）。
        """
        from concurrent.futures import ProcessPoolExecutor
        import tqdm as _tqdm
        jobs = [(str(info['cut_onnx']), str(info['ov_json']), info['layout'],
                 str(info['dlc_q'])) for info in infos]
        codes, tails = [], []
        with ProcessPoolExecutor(max_workers=self._WORKERS['converter']) as pool:
            with _tqdm.tqdm(total=len(jobs), desc='single[trunc 1/3 converter]',
                            unit='it', ncols=110, file=sys.stderr) as bar:
                for rc, tail in pool.map(_trunc_conv_one, jobs, chunksize=1):
                    codes.append(rc)
                    tails.append(tail)
                    bar.update(1)
        conv_ok = [info for info, c in zip(infos, codes) if c == 0]
        n_fail = len(infos) - len(conv_ok)
        if n_fail:
            print(f"[QAIRTAccuracyDebugger] truncated converter failures: {n_fail}")
            for info, rc, tail in zip(infos, codes, tails):
                if rc != 0 and tail:
                    print(f"  [conv fail] {info['tensor']}: {tail.strip()[-600:]}")
        return conv_ok

    def _stage_meta(self, conv_ok: list) -> list:
        """阶段B：tiny DLC meta（主进程直读 qti，带 tqdm；单文件失败不拖累其余）。

        返回与 conv_ok 对齐的 [(info, qmeta), ...]（meta 读取失败的项剔除）。
        """
        metas = self.query_tiny_dlc_batch([info['dlc_q'] for info in conv_ok])
        meta_ok = [(info, qm) for info, qm in zip(conv_ok, metas) if qm]
        print(f"[QAIRTAccuracyDebugger] truncated meta ok: "
              f"{len(meta_ok)}/{len(conv_ok)} (in-process)")
        return meta_ok

    def _stage_context_binary(self, meta_ok: list) -> list:
        """阶段C：ctx binary（并行，子进程输出静默）。

        混合精度 DLC（目标层量化 + 其余 fp16 fallback）含 Convert 节点，
        qnn-net-run 直接 --dlc_path 会在 composeGraphs 阶段报
        "QNN_DEFINITION_IMPL_GENERATED tensor not supported as output for
        Convert"；必须先离线编译成 context binary，再 --retrieve_context 执行。
        """
        def _ctx(item):
            info, _qm = item
            return self.run_qairt_context_binary_generator(
                str(info['dlc_q']), str(info['sdir'] / 'quant.bin'),
                str(info['sdir']), print_output=False)

        MultThreadExetutor.set_max_workers(self._WORKERS['ctx'])
        ctx_items = [((item,), {}) for item in meta_ok]
        ctx_codes = self._run_batch_progress(_ctx, ctx_items,
                                            'single[trunc 2/3 ctx]')
        ctx_ok = [item for item, c in zip(meta_ok, ctx_codes) if c == 0]
        return ctx_ok

    def _stage_feed_and_netrun(self, ctx_ok: list, golden: dict,
                               dims_nchw: dict) -> list:
        """阶段D（写 fp32 feed）+ 阶段E（net-run --retrieve_context），并行。

        feed 规则：输入文件永远是 fp32（net-run 对定点点输入自行量化，不要
        预量化）；模型原始输入喂 input_list 对应 raw（顺序按 set_input_order）；
        中间激活喂 golden dump（NHWC）按 tiny-DLC 外部 dims 重排。
        """
        dbg = self.dbg

        # ---- 阶段D：feed（并行）----
        def _feed(item):
            info, qm = item
            return self._trunc_write_feed(info, qm, golden, dims_nchw)

        MultThreadExetutor.set_max_workers(self._WORKERS['feed'])
        feed_items = [((item,), {}) for item in ctx_ok]
        ils = self._run_batch_progress(_feed, feed_items,
                                      'single[trunc feeds]')
        ready = [(item, il) for item, il in zip(ctx_ok, ils) if il]

        # ---- 阶段E：net-run（并行）----
        def _net(item):
            (info, _qm), il = item
            return self.run_qnn_net_run_bin(
                str(info['sdir'] / 'quant.bin'), dbg.backend_lib_target,
                il, str(info['sdir'] / 'quant_run'), print_output=False)

        MultThreadExetutor.set_max_workers(self._WORKERS['net_run'])
        net_items = [((item,), {}) for item in ready]
        net_codes = self._run_batch_progress(_net, net_items,
                                            'single[trunc 3/3 net-run]')
        net_ok = [item for item, c in zip(ready, net_codes) if c == 0]
        return net_ok

    def _collect_results(self, net_ok: list, golden: dict, dims_nchw: dict,
                         total: int) -> tuple:
        """阶段8：best-match 比对（扫 dump 文件 × golden 两种布局取最大 cos）。

        无任何成功层时抛错；成功数/候选总数（total）打印在汇报里。
        """
        accuracy = {}
        for (info, qm), _il in net_ok:
            r = self._trunc_compare(info, qm, golden, dims_nchw)
            if r is None:
                continue
            accuracy[r['name']] = {
                'entire_cos': None, 'entire_euc': None, 'entire_mse': None,
                'single_cos': r['cos'], 'single_euc': r['euc'],
            }
        if not accuracy:
            raise RuntimeError("[QAIRTAccuracyDebugger] truncated: no layer succeeded")
        print(f"[QAIRTAccuracyDebugger] single(truncated) succeeded: "
              f"{len(accuracy)}/{total}")
        return accuracy

    # ------------------------------------------------------------------
    # 单层内部逻辑：准备 / 布局 / 喂数 / 比对（info 字典见 _trunc_prepare_layer）
    # ------------------------------------------------------------------

    def _trunc_prepare_layer(self, sdir: Path, layer: dict, v2_encodings_path: str,
                             dims_nchw: dict) -> dict | None:
        """单层准备：切 ONNX（保留前 N_BACK 层浮点上界 + 边界安全化）+
        生成单层 override json。

        不可截断（无激活输入 / 输出不在 ONNX 图中 / 边界仍不安全）返回 None。
        """
        tensor = layer['tensor']
        try:
            graph = onnx.load(str(self.onnx_path)).graph
        except Exception as exc:
            print(f"[QAIRTAccuracyDebugger] onnx load failed ({exc})")
            return None
        acts0 = self.derive_activation_inputs(graph, layer)
        if not acts0:
            print(f"[QAIRTAccuracyDebugger] skip (no activation input): {tensor}")
            return None
        if tensor not in {o for n in graph.node for o in n.output}:
            print(f"[QAIRTAccuracyDebugger] skip (not an onnx output): {tensor}")
            return None
        acts = self.extend_boundary_inputs(graph, acts0, n_back=self.N_BACK)
        acts = self.extend_boundary_inputs_safe(graph, acts, tensor)
        unsafe = self.boundary_unsafe_inputs(graph, acts, tensor)
        if unsafe:
            # 兜底（宁缺勿假）：无法再向上扩（已到图输入/分叉）时跳过该层，
            # 避免边界布局二次置换导致的静默错值进入 CSV。
            print(f"[QAIRTAccuracyDebugger] skip (unsafe boundary, cannot extend): "
                  f"{tensor} <- {unsafe}")
            return None
        cut_onnx = sdir / 'cut.onnx'
        try:
            self.build_cut_onnx(str(self.onnx_path), cut_onnx, acts, tensor)
        except Exception as exc:
            print(f"[QAIRTAccuracyDebugger] cut failed for {tensor}: {type(exc).__name__}")
            return None
        ov_json = sdir / 'overrides.json'
        self.make_layer_override_json(layer['io_tensors'], v2_encodings_path,
                                      str(ov_json))
        return {
            'tensor': tensor, 'act_inputs': acts, 'sdir': sdir,
            'cut_onnx': cut_onnx, 'ov_json': ov_json,
            'layout': self._trunc_layout_args(acts),
            'dlc_q': sdir / 'quant.dlc',
        }

    def _trunc_layout_args(self, acts: list) -> str:
        # 模型原始输入（不限于名为 input0；凡 ONNX graph.input，如多输入模型）
        # 的 source 跟随 set_input_order：raw 是 nchw 还是 nhwc 由
        # prepare_input_data 决定，converter 按声明的 source 解释外部输入；
        # 内部统一 desired NHWC（与整图/激活 golden dump 一致）。
        # 其余激活 cut 输入保持 ONNX 自然 NCHW -> 内部 NHWC（喂数按 meta dims 重排）。
        io = getattr(self, '_input_order', 'nhwc')
        src_upper = 'NCHW' if str(io).lower() == 'nchw' else 'NHWC'
        model_ins = getattr(self, '_model_input_names', None) or []
        args = ''
        for a in acts:
            src = src_upper if a in model_ins else 'NCHW'
            args += (f' --source_model_input_layout "{a}" {src}'
                     f' --desired_input_layout "{a}" NHWC')
        return args

    def _trunc_write_feed(self, info: dict, qmeta: dict, golden: dict,
                          dims_nchw: dict) -> str | None:
        """按 tiny DLC 图输入顺序写 fp32 feed（input_list 路径返回给 net-run）。

        模型原始输入（不限于 input0）：raw 来自 input_list 的对应文件，顺序由
        set_input_order 决定，再按 tiny-DLC 外部 dims 重排；
        中间激活：golden dump（NHWC）按 tiny-DLC 外部 dims 重排。
        """
        try:
            model_ins = getattr(self, '_model_input_names', None) or []
            raw_by = getattr(self, '_input_raw_by_name', {})
            feed_dir = info['sdir'] / 'feed'
            feed_dir.mkdir(parents=True, exist_ok=True)
            raws = []
            for i, mi in enumerate(qmeta['inputs']):
                name = mi['name']
                if name in model_ins:
                    data = np.fromfile(raw_by[name], dtype=np.float32)
                    src_order = 'nchw' if getattr(self, '_input_order', 'nhwc') == 'nchw' else 'nhwc'
                    dst = self.trunc_order_of(mi['dims'], dims_nchw.get(name))
                    if dst is not None and dst != src_order:
                        data = self.trunc_arrange(data, src_order, dst,
                                                  dims_nchw.get(name))
                else:
                    key = self._sanitize_name(name)
                    if key not in golden:
                        raise KeyError(f'golden dump missing for input tensor {name!r}')
                    data = golden[key].astype(np.float32)
                    dst = self.trunc_order_of(mi['dims'], dims_nchw.get(name))
                    if dst == 'nchw':
                        data = self.trunc_arrange(data, 'nhwc', 'nchw', dims_nchw.get(name))
                p = feed_dir / f'in{i}_{self._sanitize_name(name)}.raw'
                data.astype(np.float32).tofile(str(p))
                raws.append(str(p))
            il = info['sdir'] / 'input_list.txt'
            il.write_text(' '.join(str(p) for p in raws) + '\n', encoding='utf-8')
            return str(il)
        except Exception as exc:
            print(f"[QAIRTAccuracyDebugger] feed failed for {info['tensor']}: "
                  f"{type(exc).__name__}: {exc}")
            return None

    def _trunc_compare(self, info: dict, qmeta: dict, golden: dict,
                       dims_nchw: dict) -> dict | None:
        """best-match dump 输出 vs golden（两种布局试，量化输出按编码反量化）。"""
        tensor = info['tensor']
        key = self._sanitize_name(tensor)
        gold = golden[key]
        out_dims = dims_nchw.get(tensor)
        refs = {'direct': gold}
        if out_dims is not None and len(out_dims) == 4:
            refs['arr'] = self.trunc_arrange(gold, 'nhwc', 'nchw', out_dims)

        def load_arr(path: Path):
            L = path.stat().st_size
            if L == 4 * gold.size:
                return np.fromfile(str(path), dtype=np.float32)
            if L == gold.size:
                tname = next((n for n in qmeta['tensors']
                              if self._sanitize_name(n) == path.stem), None)
                te = qmeta['tensors'].get(tname, {}) if tname else {}
                s = te.get('scale')
                if s is None:
                    return None
                z = te.get('zp', 0)
                dt = np.int8 if te.get('dtype') == 'int8' else np.uint8
                q = np.fromfile(str(path), dtype=dt)
                return ((q.astype(np.float64) + z) * s).astype(np.float32)
            return None

        out_dir = info['sdir'] / 'quant_run'
        best = (-2.0, None, None)
        for p in Path(out_dir).rglob('*.raw'):
            arr = load_arr(p)
            if arr is None:
                continue
            for _mode, gref in refs.items():
                if arr.size != gref.size:
                    continue
                c = float(np.dot(gref, arr) / (np.linalg.norm(gref) * np.linalg.norm(arr) + 1e-12))
                if c > best[0]:
                    best = (c, float(np.linalg.norm(gref - arr)), p.name)
        if best[0] < -1.5:
            print(f"[QAIRTAccuracyDebugger] output missing for {tensor}")
            return None
        return {'name': key, 'cos': best[0], 'euc': best[1],
                'mse': float(best[1] * best[1] / gold.size)}

    # ------------------------------------------------------------------
    # golden dump / tiny-DLC meta 读取（主进程直读，不递归子进程）
    # ------------------------------------------------------------------

    def query_tiny_dlc_batch(self, dlc_paths: list) -> list:
        """批量读取多个 tiny DLC 的图输入顺序/量化参数（主进程直读）。

        每个文件新建 qti reader 读取（复用 reader 会串数据）；单个失败不
        拖累其余，返回与 dlc_paths 对齐的列表（失败元素为 None）。
        """
        import tqdm
        if not dlc_paths:
            return []
        out = []
        with tqdm.tqdm(total=len(dlc_paths), desc='single[trunc meta]', unit='it',
                       ncols=110, file=sys.stderr) as bar:
            for p in dlc_paths:
                try:
                    out.append(self.query_tiny_dlc(p))
                except Exception as exc:
                    print(f"[QAIRTAccuracyDebugger] tinydlc failed {p}: "
                          f"{type(exc).__name__}: {exc}")
                    out.append(None)
                bar.update(1)
        return out

    def query_tiny_dlc(self, dlc_path) -> dict:
        """读取单个 tiny DLC 图输入顺序/输出/张量量化参数（并落盘 meta json）。"""
        meta = _read_tiny_dlc_meta(str(dlc_path))
        with open(Path(dlc_path).with_suffix('.tiny_meta.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        return meta

    # ------------------------------------------------------------------
    # SDK 工具（自带副本；混合精度
    # DLC 须先编译 ctx binary 再 --retrieve_context 执行，原因见上方阶段C）
    # ------------------------------------------------------------------

    def run_qairt_context_binary_generator(self, dlc_path: str, bin_path: str, output_dir: str,
                                           output_tensors: str = None, graph_name: str = None,
                                           print_output: bool = True) -> int:
        """qnn-context-binary-generator --dlc_path <半量化DLC> -> context binary (.bin)。

        Args:
            dlc_path: 半量化 DLC（converter 产物）。
            bin_path: 输出的 context binary 路径（.bin）。
            output_dir: 工具输出目录（也放 binary_file 的同级默认位置）。
            output_tensors: 要额外导出的中间张量名（逗号分隔，不带 graph 前缀）。
            graph_name: DLC 中的 graph 名（如 "RetinaFace_mobile_1_3_320_320"）。
                        --set_output_tensors 需 "graph:tensor" 语法；为 None 时只传张量名。
        """
        # qnn-context-binary-generator 的输出文件名 = --binary_file 参数值 + ".bin"
        # （实测:传 layer -> layer.bin;传 layer.bin -> layer.bin.bin）。
        # 因此去掉 .bin 后缀,最终产物路径 = bin_path 本身。

        # model, backend
        if sys.platform.startswith('win'):
            model_lib, backend_lib = 'QnnModelDlc.dll', 'QnnHtp.dll'
        else:
            model_lib, backend_lib = 'libQnnModelDlc.so', 'libQnnHtp.so'

        bin_stem = bin_path[:-4] if bin_path.endswith('.bin') else bin_path

        cmd = f'qnn-context-binary-generator --model {model_lib} --backend {backend_lib} --log_level error'
        cmd += f' --dlc_path {dlc_path} --binary_file {bin_stem} --output_dir {output_dir}'
        if output_tensors:
            if graph_name:
                cmd += f' --set_output_tensors "{graph_name}:{output_tensors}"'
            else:
                cmd += f' --set_output_tensors "{output_tensors}"'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]", print_output=print_output)

    def run_qnn_net_run_bin(self, bin_path: str, backend_lib: str, input_list: str,
                            output_dir: str, print_output: bool = True) -> int:
        """执行 qnn-net-run --retrieve_context <bin>（context binary 加载执行）。"""
        cmd = f'qnn-net-run --retrieve_context {bin_path} --backend {backend_lib}'
        cmd += f' --input_list {input_list} --output_dir {output_dir} --log_level error'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]", print_output=print_output)

    def extract_dlc_encoding(self, model_path: str) -> str:
        """量化 DLC -> v2.0.0 encodings json（主进程直接读，不递归子进程）。

        需 SDK 环境已注入 os.environ（run_env_script / setup_sdk），qti 在该
        进程内可导入。
        """
        self.dbg.working_dir.mkdir(parents=True, exist_ok=True)
        encoding_path = self.dbg.working_dir / f"{Path(model_path).stem}_encoding.json"
        DlcV2EncodingExporter(model_path).dump(str(encoding_path))
        return str(encoding_path)

    def _sanitize_name(self, name: str) -> str:
        """张量名 -> 输出 raw 文件名（qnn-net-run: 非字母数字下划线统一替换为 '_'）。"""
        return re.sub(r'[^A-Za-z0-9_]', '_', name)

    def extract_layer_candidates(self, quantized_dlc_path: str) -> list[dict]:
        """遍历量化 DLC 图，为每个可量化 op 产出层候选（主进程直接读）。

        每个候选 = {"tensor": <该层输出张量名>, "op_type": <op类型>,
                    "io_tensors": [op 输入+输出张量名...]}。
        与官方 subgraph_snooper 的 overrides 范围一致：
        Conv2d -> input+weight+bias+output（4个）；Prelu -> input+coeff+output（3个）；
        Concat -> 各输入激活+output（4个）。
        """
        return DlcV2EncodingExporter(quantized_dlc_path).layer_candidates()

    def make_layer_override_json(self, io_tensor_names: list, v2_encodings_path: str,
                                 out_json_path: str) -> str:
        """按张量名从全量 v2 encodings 中筛出子集，生成单层 quantization_overrides json。

        Args:
            io_tensor_names: 该层 producer op 的输入+输出张量名。
            v2_encodings_path: 全量化 DLC 导出的 v2.0.0 encodings json。
            out_json_path: 输出的 overrides json 路径（AIMET 2.0.0 格式）。
        """
        with open(v2_encodings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        encodings = data.get('encodings', data if isinstance(data, list) else [])
        by_name = {e.get('name'): e for e in encodings}
        want = [n for n in io_tensor_names if n in by_name]

        # y_zero_point 符号：AIMET 2.0.0 overrides 约定为正 offset（AIMET: Positive,
        # QNN: Negative），量化 DLC 直接导出的是 QNN 负值，需取反后才能喂 converter。
        def flip_zp(v):
            if isinstance(v, list):
                return [-1.0 * x for x in v]
            return -1.0 * v

        out_encs = []
        for e in (by_name[n] for n in want):
            e = dict(e)
            if 'y_zero_point' in e:
                e['y_zero_point'] = flip_zp(e['y_zero_point'])
            out_encs.append(e)
        out = {'version': '2.0.0', 'encodings': out_encs}
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)
        return out_json_path

    def _run_batch_progress(self, fn, items, desc) -> list:
        """并行提交一批任务并显示 tqdm 进度条，返回与 items 顺序一致的返回码。

        items: [(args_tuple, kwargs_dict), ...]；每个任务经 MultThreadExetutor
        线程池执行。子进程输出在调用方以 print_output=False 静默，终端只保留
        进度条。
        """
        from concurrent.futures import as_completed
        import tqdm

        if not items:
            return []  # 空批次(如前面阶段全部失败)：直接返回，不闪 0 层进度条

        for args, kw in items:
            MultThreadExetutor.run_exetutor(fn, *args, **kw)
        n = len(items)
        idx_of = {id(f): i for i, f in enumerate(MultThreadExetutor.future_list)}
        codes = [None] * n
        with tqdm.tqdm(total=n, desc=desc, unit='it', ncols=110, file=sys.stderr) as bar:
            for fut in as_completed(MultThreadExetutor.future_list):
                i = idx_of[id(fut)]
                codes[i] = fut.result()
                bar.update(1)
        MultThreadExetutor.wait_and_close()  # 收尾并重置线程池供下一阶段
        return codes

    # ------------------------------------------------------------------
    # 静态纯函数工具（无实例状态；布局重排 / ONNX 切图 / tiny meta 导出）
    # ------------------------------------------------------------------

    @staticmethod
    def onnx_graph_dims_nchw(onnx_path: str) -> dict:
        """ONNX 模型内 rank-4 张量的 NCHW dims（截断喂数/比对布局用）。"""
        m = onnx.load(onnx_path)
        dims = {}
        for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            tt = v.type.tensor_type
            if not tt.HasField('shape'):
                continue
            d = [x.dim_value if x.HasField('dim_value') else None for x in tt.shape.dim]
            if len(d) == 4 and all(x is not None for x in d):
                dims[v.name] = d
        return dims

    @staticmethod
    def trunc_perm_nhwc(nchw_dims: list) -> list:
        return [nchw_dims[0], nchw_dims[2], nchw_dims[3], nchw_dims[1]]

    @classmethod
    def trunc_arrange(cls, flat: np.ndarray, src_order: str, dst_order: str,
                      nchw_dims: list | None) -> np.ndarray:
        """flat fp32 张量在 'nchw'/'nhwc' 间重排（仅 rank-4）。"""
        if nchw_dims is None or len(nchw_dims) != 4 or src_order == dst_order:
            return flat
        if src_order == 'nhwc' and dst_order == 'nchw':
            return flat.reshape(cls.trunc_perm_nhwc(nchw_dims)).transpose(0, 3, 1, 2).reshape(-1)
        if src_order == 'nchw' and dst_order == 'nhwc':
            return flat.reshape(nchw_dims).transpose(0, 2, 3, 1).reshape(-1)
        return flat

    @classmethod
    def trunc_order_of(cls, dims: list, nchw_dims: list | None) -> str | None:
        if nchw_dims is None or len(nchw_dims) != 4 or len(dims) != 4:
            return None
        if list(dims) == list(nchw_dims):
            return 'nchw'
        if list(dims) == cls.trunc_perm_nhwc(nchw_dims):
            return 'nhwc'
        return None

    @staticmethod
    def derive_activation_inputs(graph, candidate: dict) -> list:
        """DLC 候选 io_tensors -> 该算子 ONNX 侧的激活输入张量名。"""
        init = {i.name for i in graph.initializer}
        known = (init | {i.name for i in graph.input} | {o.name for o in graph.output}
                 | {v.name for v in graph.value_info}
                 | {o for n in graph.node for o in n.output})
        acts = []
        for t in candidate['io_tensors'][:-1]:          # 去掉算子输出（最后一个）
            if t in init:                                # 权重/偏置 initializer
                continue
            if t not in known:                           # DLC-only（converter 生成）
                continue
            acts.append(t)
        return acts

    @staticmethod
    def extend_boundary_inputs(graph, start_inputs: list, n_back: int = 2) -> list:
        """把截断边界再往上游扩 n_back 层算子（这些层保持浮点，使目标层量化
        输入由图内 float->int 边界产生——与可用的整图半量化结构一致）。"""
        producer = {}
        for n in graph.node:
            for o in n.output:
                producer[o] = n
        graph_in = {i.name for i in graph.input}
        init = {i.name for i in graph.initializer}

        def act_inputs_of(node):
            return [i for i in node.input
                    if i and i not in init and (i in producer or i in graph_in)]

        cur = list(dict.fromkeys(start_inputs))
        for _ in range(n_back):
            nxt = []
            changed = False
            for t in cur:
                node = producer.get(t)
                if node is None:
                    nxt.append(t)
                    continue
                ai = act_inputs_of(node)
                if len(node.output) == 1 and len(ai) == 1 and ai[0] not in cur:
                    nxt.append(ai[0])
                    changed = True
                else:
                    nxt.append(t)
            if not changed:
                break
            cur = list(dict.fromkeys(nxt))
        return cur

    @staticmethod
    def boundary_unsafe_inputs(graph, act_inputs: list, target_tensor: str) -> list:
        """返回边界中"被非目标 shape 算子直接消费"的张量名（空 = 安全）。

        目标算子自身是 shape 算子时不算不安全：小图从目标算子开始，其输入
        顺序与整图一致（实测 N_BACK=2 的 9 个 Reshape 层数值正确）。
        """
        shape_ops = QnnTruncatedAccuracyAnalysis._BOUNDARY_SHAPE_OPS
        consumers = {}
        for n in graph.node:
            for i in n.input:
                consumers.setdefault(i, []).append(n)
        return [a for a in act_inputs
                for c in consumers.get(a, [])
                if target_tensor not in c.output and c.op_type in shape_ops]

    @staticmethod
    def extend_boundary_inputs_safe(graph, act_inputs: list, target_tensor: str,
                                    max_extra: int = None) -> list:
        """边界安全化：边界张量不得被"非目标"的 shape 算子直接消费。

        命中时继续向上游扩一层（复用 extend_boundary_inputs 的单步语义），
        直到边界进入点是 Conv/Pool 等布局规范算子或目标算子本身；无法再扩
        （已到图输入/分叉）时原样返回，由 boundary_unsafe_inputs 兜底。
        """
        if max_extra is None:
            max_extra = QnnTruncatedAccuracyAnalysis.MAX_BOUNDARY_EXTEND
        acts = list(act_inputs)
        for _ in range(max_extra):
            if not QnnTruncatedAccuracyAnalysis.boundary_unsafe_inputs(
                    graph, acts, target_tensor):
                break
            nxt = QnnTruncatedAccuracyAnalysis.extend_boundary_inputs(
                graph, acts, n_back=1)
            if nxt == acts:
                break
            acts = nxt
        return acts

    @staticmethod
    def build_cut_onnx(onnx_path: str, out_onnx: Path, act_inputs: list,
                       out_tensor: str) -> None:
        """从完整 ONNX 切出 [act_inputs -> out_tensor] 子图（张量名保留）。"""
        m = onnx.load(onnx_path)
        g = m.graph
        init_names = {i.name for i in g.initializer}

        producer = {}
        for n in g.node:
            for o in n.output:
                producer[o] = n
        if out_tensor not in producer:
            raise KeyError(f'output tensor {out_tensor!r} not produced by any ONNX node')

        boundary = set(act_inputs)
        needed = set()
        queue = [out_tensor]
        while queue:
            t = queue.pop(0)
            node = producer.get(t)
            if node is None:
                continue
            nid = id(node)
            if nid in needed:
                continue
            needed.add(nid)
            for it in node.input:
                if it and it not in boundary and it not in init_names and it in producer:
                    queue.append(it)

        keep = [n for n in g.node if id(n) in needed]

        dims_of = {}
        for v in list(g.value_info) + list(g.input) + list(g.output):
            tt = v.type.tensor_type
            if tt.HasField('shape'):
                dims_of[v.name] = [d.dim_value if d.HasField('dim_value') else None
                                   for d in tt.shape.dim]

        def vi_proto(name: str):
            dims = dims_of.get(name)
            if dims is None:
                return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            return helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [d if d is not None else 1 for d in dims])

        need_init = sorted({it for n in keep for it in n.input if it in init_names})
        init_by_name = {i.name: i for i in g.initializer}

        new_graph = helper.make_graph(
            keep, 'trunc',
            [vi_proto(a) for a in act_inputs],
            [vi_proto(out_tensor)],
            [init_by_name[x] for x in need_init])
        nm = helper.make_model(new_graph, opset_imports=list(m.opset_import))
        nm.ir_version = m.ir_version
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(nm, str(out_onnx))

    @staticmethod
    def dump_tinydlc_meta(dlc_path: str, out_json: str) -> str:
        """tiny DLC -> {inputs/outputs/tensors(+量化参数)} json（CLI: tinydlc）。"""
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(_read_tiny_dlc_meta(dlc_path), f, indent=1)
        return out_json

    @staticmethod
    def dump_tinydlc_meta_batch(dlcs_json: str) -> int:
        """批量读取多个 tiny DLC 的图信息（一次 python 进程，CLI: tinydlc-batch）。

        dlcs_json: {"dlcs": [<dlc 绝对路径>...]}；每个 dlc 的 meta 写到
        <dlc 同目录>/<stem>.tiny_meta.json（与单文件 tinydlc 一致）。
        """
        with open(dlcs_json, 'r', encoding='utf-8') as f:
            dlc_list = json.load(f)['dlcs']
        for dlc in dlc_list:
            try:
                _write_tiny_dlc_meta(dlc)
            except Exception as exc:
                print(f'[tinydlc-batch] failed {dlc}: {type(exc).__name__}: {exc}')
        return 0

class AccuracyGraph:
    """
    精度分析网络图（Netron 风格）。

    将 RKNN/QNN 快照层以"方框节点 + 箭头连线"呈现为从左到右分层的 DAG，
    节点填充色按累积精度 entire_cos 着色（红=差, 绿=良），悬停显示
    单层精度与欧氏距离等详细信息。

    Attributes:
        data (dict): AccuracyGraph 内部数据结构（rows/inputs/outputs/paths）。
        children (dict[str, list[str]]): 层 -> 下游层列表。
        parents (dict[str, list[str]]): 层 -> 上游层列表。
    """

    # 常见算子类型 -> 边框颜色（Netron 风格色板）
    # Netron (lutzroeder/netron) 类别色板（grapher.css）
    CATEGORY_COLORS: dict[str, str] = {
        'Layer': '#335588',          # 卷积、全连接等
        'Activation': '#702921',     # ReLU、Sigmoid、Softmax、Clip 等
        'Pool': '#335533',           # 池化
        'Normalization': '#335544',  # LayerNorm、BatchNorm 等
        'Dropout': '#454770',
        'Shape': '#6c4f47',          # Reshape、Flatten 等
        'Tensor': '#59423b',         # Concat、Split 等
        'Transform': '#335544',      # Transpose、Squeeze、Unsqueeze 等
        'Data': '#555555',           # 输入/输出数据
        'Quantization': '#502800',   # 量化层
        'Attention': '#783c00',      # 注意力
        'Constant': '#eeeeee',       # 常量
        'Control': '#eeeeee',        # 控制流
    }
    DEFAULT_COLOR = '#333333'        # Netron 未分类默认色

    # 算子类型 -> Netron 类别
    OP_CATEGORY: dict[str, str] = {
        'Conv': 'Layer', 'Gemm': 'Layer', 'MatMul': 'Layer',
        'Add': 'Layer', 'Mul': 'Layer', 'Sub': 'Layer', 'Div': 'Layer',
        'Concat': 'Tensor', 'Split': 'Tensor', 'Slice': 'Tensor',
        'Reshape': 'Shape', 'Flatten': 'Shape',
        'Transpose': 'Transform', 'Squeeze': 'Transform', 'Unsqueeze': 'Transform',
        'Softmax': 'Activation', 'Sigmoid': 'Activation', 'Relu': 'Activation',
        'LeakyRelu': 'Activation', 'Clip': 'Activation', 'exSwish': 'Activation',
        'LayerNormalization': 'Normalization', 'LayerNorm': 'Normalization',
        'exNorm': 'Normalization', 'BatchNormalization': 'Normalization',
        'exDataConvert': 'Data', 'Input': 'Data', 'Output': 'Data',
        'exSDPAttention': 'Attention',
        # ---- QNN (DLC 反射 op_type，见 docs/.../SupportedOps.html) ----
        # 常用算子：保持精简，按实际模型出现与常见 QNN 算子补齐
        # ---- QNN (DLC 反射 op_type，见 docs/.../SupportedOps.html) ----
        # 只列前段(RKNN)未出现的 QNN 算子；同名映射已在上面定义，避免重复键
        'Conv2d': 'Layer', 'DepthWiseConv2d': 'Layer', 'FullyConnected': 'Layer',
        'TransposeConv2d': 'Layer',
        'Eltwise_Binary': 'Layer', 'Eltwise_Or': 'Layer',
        'Eltwise_And': 'Layer', 'Eltwise_Not': 'Layer',
        'Prelu': 'Activation', 'ReluMinMax': 'Activation',
        'Tanh': 'Activation', 'Gelu': 'Activation', 'HardSwish': 'Activation',
        'ElementWiseNeuron': 'Activation', 'ElementWiseUnary': 'Activation',
        'Resize': 'Transform',   # QNN Resize (最近邻/双线性上采样)
        'PoolMax': 'Pool', 'PoolAvg': 'Pool', 'Pooling': 'Pool',
        'BatchNorm': 'Normalization', 'RmsNorm': 'Normalization',
        'InstanceNorm': 'Normalization',
    }

    def __init__(
        self,
        accuracy: dict,
        onnx_path: str | Path,
        output_path: str | Path,
        title: str = 'Accuracy Analysis',
        *,
        sanitize: bool = False,
        include_input: bool = False,
        layer_display=None,
        dlc_candidates: list[dict] | None = None,
        dlc_io: dict | None = None,
    ):
        """基于统一"层精度" dict 构建精度网络图（Netron 风格 HTML）。

        onnx 计算图在本构造器内构建一次（_parse_accuracy_dict），不再由
        RknnAccuracyDebugger / QnnAccuracyDebugger 各自构建（消除重复 load）。

        Args:
            accuracy: 统一层精度 dict：{层名: {'entire_cos'|'entire_euc'|'entire_mse'|
                        'single_cos'|'single_euc': float|None, 'op_type': str|None(可选)}}。
            onnx_path: 用于图结构/拓扑/算子类型推断的 ONNX 模型。
            output_path: 输出 HTML 路径。
            title: 图标题（如 'RKNN Graph Accuracy Analysis' / 'QNN ...'）。
            sanitize: 层名是否为 sanitized（QNN raw 文件名清洗名 -> True；
                      RKNN 快照名基于原始 ONNX 张量名 -> False）。
            include_input: 是否追加 Input 示意终端（QNN raw 不含输入层 -> True；
                           RKNN 快照已含输入层 -> False）。
            layer_display: (graph, key) -> 节点显示名 的 resolver；graph 为 _parse
                           内部构建好的 onnx 图（build_onnx_tensor_graph 返回值），
                           调用方无需自行再 load。None = 恒等（dict 键即显示名，RKNN
                           用）；QNN 传 lambda g, k: g['tensors'].get(k,k) 把 sanitized
                           键还原成原始 ONNX 张量名。
            dlc_candidates: 可选，量化 DLC 反射出的层候选列表（DlcV2EncodingExporter.
                           layer_candidates() 产物：每项 {tensor, op_type, io_tensors}）。
                           提供时本图按 **DLC 视角** 构建：
                           - 节点全集 = 全部候选 op（精度结果没有的层也画，精度 None）；
                           - 节点 op_type = DLC 反射类型（如 Prelu/Conv2d/Eltwise_Binary）；
                           - 连线 = 按候选 io_tensors（DLC 实际拓扑，含 converter 融合层）。
                           None（默认）时维持 ONNX 视角（RKNN / 无 DLC 时用）。
            dlc_io: 可选，DLC IR 图边界 {"inputs": [...], "outputs": [...]}
                           （DlcV2EncodingExporter.graph_io_tensors() 产物）。
                           提供时 Input/Output 终端直接取图边界；None 时按
                           "未被任何候选产出/消费"的启发式反推（旧行为）。
        """
        self._accuracy = accuracy or {}
        self.onnx_path = Path(onnx_path)
        self._sanitize = sanitize
        self._include_input = include_input
        self._layer_display = layer_display
        self._dlc_candidates = dlc_candidates or None
        self._dlc_io = dlc_io or None

        self.output_path = Path(output_path)
        self.title = title
        self.node_sep = 50.0 # 横向：真实节点之间的间距
        self.edge_sep = 20.0 # 横向：长边虚拟节点占用的间距
        self.rank_sep = 50.0 # 纵向：层与层之间的间距
        self.pan_speed:float = 15

        self.rows: list[dict] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.node_order: dict[str, int] = {}
        self.layer_row: dict[str, dict] = {}
        self.euc_color_min = 0.0
        self.euc_color_max = 1.0

        self._parse_accuracy_dict()    # onnx 构建 + rows/终端/层图推导

    # ------------------------------------------------------------------
    # 内部：由统一 dict 构建 onnx 图 / 层图 / 终端节点
    # ------------------------------------------------------------------

    def _parse_accuracy_dict(self) -> None:
        """从统一"层精度" dict 推导本图所需的全部数据（在 __init__ 调用）。

        两种模式（构造器 dlc_candidates 决定）：
          A) DLC 模式（QNN，dlc_candidates 非 None）：
             节点全集 = 量化 DLC 的全部层候选（精度结果缺失的层也画，数值 None）；
             op_type = DLC 反射类型（Prelu/Conv2d/Eltwise_Binary/...，与 encoding 一致）；
             连线 = 按候选 io_tensors（DLC 实际拓扑，含 converter 融合/自造张量）。
          B) ONNX 模式（RKNN / 无 DLC）：
             onnx 计算图构建一次；层全集 = accuracy dict 键；op_type = dict 自带或
             onnx node_info 回退；连线走 onnx 张量 succ。
        两种模式最后都产出 rows/children/parents/inputs/outputs 及布局所需字段。
        """
        if self._dlc_candidates:
            self._parse_accuracy_from_dlc()
        else:
            self._parse_accuracy_from_onnx()

    def _finish_parse(self, real_rows: list[dict], children: dict, parents: dict,
                      input_rows: list[dict], output_rows: list[dict],
                      input_layers: list[str], output_layers: list[str]) -> None:
        """收尾：组装 self.rows / children / parents / 颜色范围（两模式共用）。"""
        self.rows = input_rows + real_rows + output_rows
        self.inputs = input_layers
        self.outputs = output_layers
        self.children = children
        self.parents = parents
        self.data = {
            'rows': self.rows,
            'inputs': self.inputs,
            'outputs': self.outputs,
        }
        self.node_order = {r['layer_name']: i for i, r in enumerate(self.rows)}
        self.layer_row = {r['layer_name']: r for r in self.rows}
        euc_values = [r.get('entire_euc') for r in self.rows
                  if r.get('entire_euc') is not None]
        self.euc_color_min = min(euc_values, default=0.0)
        self.euc_color_max = max(euc_values, default=1.0)

    # ------------------------------------------------------------------
    # B) ONNX 模式（默认）：层全集 = accuracy 键；连线走 onnx 张量 succ
    # ------------------------------------------------------------------

    def _parse_accuracy_from_onnx(self) -> None:
        graph = build_onnx_tensor_graph(self.onnx_path, sanitize=self._sanitize)
        node_info = graph['node_info']
        # layer_display(graph, key) -> 显示名；None 用恒等（graph 只此一次 load）
        name_of = ((lambda k: self._layer_display(graph, k))
                   if self._layer_display is not None else (lambda k: k))

        results = []
        for name, row in self._accuracy.items():
            raw_stem = str(name)
            if raw_stem.endswith('.raw'):
                raw_stem = raw_stem[:-4]
            tensor = match_tensor(raw_stem, graph)
            # op_type：优先 dict 自带（Rknn error_analysis），缺则 onnx node_info 推断
            op_type = row.get('op_type')
            if not op_type and tensor is not None and tensor in node_info:
                op_type = node_info[tensor][0]
            results.append({
                'golden': str(name), 'infer': None, 'tensor': tensor,
                'layer_name': name_of(raw_stem),
                'op_type': op_type or 'Unknown',
                'entire_cos': row.get('entire_cos'),
                'entire_euc': row.get('entire_euc'),
                'single_cos': row.get('single_cos'),
                'single_euc': row.get('single_euc'),
                'mse': row.get('entire_mse'),
            })

        real_rows = [{
            'layer_name': r['layer_name'], 'op_type': r['op_type'],
            'entire_cos': r['entire_cos'], 'entire_euc': r['entire_euc'],
            'single_cos': r['single_cos'], 'single_euc': r['single_euc'],
        } for r in results]

        children, parents, tensor_layers = build_layer_graph(results, graph)

        (input_rows, output_rows, input_layers, output_layers,
         aug_children, aug_parents) = build_terminal_nodes(
            results, children, parents, graph, include_input=self._include_input,
            name_of=name_of,
        )

        self._finish_parse(real_rows, aug_children, aug_parents,
                           input_rows, output_rows, input_layers, output_layers)

    # ------------------------------------------------------------------
    # A) DLC 模式（QNN）：节点全集 = DLC 候选层；连线走候选 io_tensors
    # ------------------------------------------------------------------

    def _parse_accuracy_from_dlc(self) -> None:
        import re as _re
        san = lambda x: _re.sub(r'[^A-Za-z0-9_]', '_', x)
        # onnx 图仍用于模型级输入/输出终端 + 显示名还原（与精度匹配分离）
        graph = build_onnx_tensor_graph(self.onnx_path, sanitize=self._sanitize)
        onnx_name_of = graph['tensors']           # sanitized 键 -> 原始 onnx 名
        name_of = ((lambda k: self._layer_display(graph, k))
                   if self._layer_display is not None else (lambda k: k))

        cands = self._dlc_candidates
        # DLC 精度表：以候选张量原始名做键（sanitize 匹配 accuracy dict 的键）
        acc_by_san = {k: row for k, row in self._accuracy.items()}

        # ---- 1) 层全集 = 全部候选 op（精度缺失层也画，None）----
        # 注意同一候选输出张量去重；DLC 输出张量 = io_tensors 的最后一项（多数 op）
        seen_tensor: set[str] = set()
        nodes: list[dict] = []                     # {name(原始tensor), op_type, acc_row|None}
        for c in cands:
            t = c['tensor']
            if t in seen_tensor:
                continue
            seen_tensor.add(t)
            acc = acc_by_san.get(san(t)) or {}
            nodes.append({
                'name': t,
                'op_type': c['op_type'],
                'entire_cos': acc.get('entire_cos'),
                'entire_euc': acc.get('entire_euc'),
                'single_cos': acc.get('single_cos'),
                'single_euc': acc.get('single_euc'),
                'mse': acc.get('entire_mse'),
            })

        # ---- 2) children/parents：producer(输出张量名) -> 消费它的候选 ----
        # io_tensors = [输入..., 输出]；输出 = 候选的 'tensor'。
        # 输入张量名若等于某候选输出 -> 连 producer -> 当前候选。
        # converter 自造/非候选张量（权重、coeff、Softmax 前的 reshape 等）作为叶子/边界。
        producer_by_out = {c['tensor']: c['tensor'] for c in cands}
        children: dict[str, list[str]] = defaultdict(list)
        parents: dict[str, list[str]] = defaultdict(list)

        for c in cands:
            cur = c['tensor']
            if len(c['io_tensors']) > 1:
                ins = c['io_tensors'][:-1]
            else:
                ins = []
            for i in ins:
                prod = producer_by_out.get(i)
                if prod and prod != cur:
                    children[prod].append(cur)
                    parents[cur].append(prod)
        # 去重 + 保持出现顺序
        for k in children:
            children[k] = list(dict.fromkeys(children[k]))
        for k in parents:
            parents[k] = list(dict.fromkeys(parents[k]))

        # ---- 3) 构造真实行（显示层名 = 原始 DLC 张量名；可带 name_of 还原）----
        real_rows = []
        for n in nodes:
            raw = n['name']
            display = name_of(raw)
            real_rows.append({
                'layer_name': display,
                'op_type': n['op_type'],
                'entire_cos': n['entire_cos'],
                'entire_euc': n['entire_euc'],
                'single_cos': n['single_cos'],
                'single_euc': n['single_euc'],
            })

        # ---- 4) children/parents 的键是原始 tensor 名；真实行 layer_name 可能被
        #        还原(显示)成不同字符串 -> 需把边键统一到 display 名 ----
        display_of = {c['tensor']: name_of(c['tensor']) for c in cands}
        def map_edge(d):
            out = defaultdict(list)
            for a, bl in d.items():
                da = display_of.get(a, a)
                for b in bl:
                    db = display_of.get(b, b)
                    if da != db:
                        out[da].append(db)
            return {k: list(dict.fromkeys(v)) for k, v in out.items()}
        children_disp = map_edge(children)
        parents_disp = map_edge(parents)

        # ---- 5) Input/Output 终端：模型级输入/输出张量 -> 连到 DLC 节点 ----
        # 用 onnx graph inputs/outputs（原始名经 sanitize 在 display 域匹配困难，
        # 直接按原始 tensor 名找出消费它的候选）。
        def first_consumer(t: str) -> str | None:
            # 在 parents(原始域)里找以 t 为输入的候选
            hit = None
            for c in cands:
                if t in c['io_tensors'][:-1] if len(c['io_tensors']) > 1 else False:
                    return display_of.get(c['tensor'], c['tensor'])
            return hit

        input_rows: list[dict] = []
        input_layers: list[str] = []
        output_rows: list[dict] = []
        output_layers: list[str] = []
        if self._include_input:
            # 图输入：优先用 DLC IR 图声明的边界（graph_io_tensors）；无该信息
            # 时退回启发式——"未被任何候选产出的候选输入"再减权重/常量名黑名单
            # （黑名单对 TF 的 `.../kernel:0`、converter hoist 常量无效，会误画
            # 成 Input 节点，故只在拿不到 IR 图边界时使用）。
            if self._dlc_io and self._dlc_io.get('inputs'):
                model_ins = [t for t in self._dlc_io['inputs']
                             if t not in producer_by_out]
            else:
                all_ins: set[str] = set()
                for c in cands:
                    if len(c['io_tensors']) > 1:
                        all_ins.update(c['io_tensors'][:-1])
                model_ins = sorted(
                    t for t in (all_ins - set(producer_by_out))
                    if not any(k in t for k in ('weight', 'bias', 'coeff', 'onnx::')))
            for inp in model_ins:
                display = inp  # 图输入没有 DLC 节点，直接显示原名
                input_layers.append(display)
                input_rows.append({
                    'layer_name': display, 'op_type': 'Input',
                    'entire_cos': 1.0, 'entire_euc': 0.0,
                    'single_cos': 1.0, 'single_euc': 0.0,
                })
                # 连到下游第一个消费该输入张量的候选
                con = first_consumer(inp)
                if con:
                    children_disp.setdefault(display, []).append(con)
                    parents_disp.setdefault(con, []).append(display)

        # 输出终端：优先用 DLC IR 图声明的图输出；无该信息时退回启发式
        # （无人消费的候选输出，如 output0/1/2）。
        # 注：output0 这类有输入(在 parents)但无下游消费者，故判 children 而非 parents
        out_set = (set(self._dlc_io['outputs'])
                   if (self._dlc_io and self._dlc_io.get('outputs')) else None)
        for c in cands:
            t = c['tensor']
            if out_set is not None:
                if t not in out_set:      # 非 IR 图输出 -> 不画 Output 终端
                    continue
            elif t in children:           # 启发式：有下游消费者 -> 非图输出
                continue
            display = display_of.get(t, t)
            if display in {r['layer_name'] for r in real_rows}:
                node_name = f'{display} (out)'
            else:
                node_name = display
            output_layers.append(node_name)
            src = [display]
            output_rows.append({
                'layer_name': node_name, 'op_type': 'Output',
                'entire_cos': next((r['entire_cos'] for r in real_rows
                                    if r['layer_name'] == display), None),
                'entire_euc': next((r['entire_euc'] for r in real_rows
                                    if r['layer_name'] == display), None),
                'single_cos': next((r['single_cos'] for r in real_rows
                                    if r['layer_name'] == display), None),
                'single_euc': next((r['single_euc'] for r in real_rows
                                    if r['layer_name'] == display), None),
            })
            for s in src:
                children_disp.setdefault(s, []).append(node_name)
                parents_disp.setdefault(node_name, []).append(s)

        self._finish_parse(real_rows, children_disp, parents_disp,
                           input_rows, output_rows, input_layers, output_layers)

    # ------------------------------------------------------------------
    # 颜色辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _cos_to_color(cos: float | None, vmin: float = 0.8, vmax: float = 1.0) -> str:
        """将累积/单层 cosine 相似度映射为红(差)-黄(中)-绿(良)。"""
        if cos is None:
            return '#b0b0b0'
        t = (cos - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        # 红(255,69,0) -> 黄(255,215,0) -> 绿(60,179,113)
        if t < 0.5:
            k = t / 0.5
            # 红/黄两端蓝色均为 0，必须保持 b=0；否则会被错误拉高成粉/橙红
            r, g, b = 255, 69 + (215 - 69) * k, 0
        else:
            k = (t - 0.5) / 0.5
            r, g, b = 255 + (60 - 255) * k, 215 + (179 - 215) * k, 0 + (113 - 0) * k
        return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

    def _op_color(self, op_type: str | None) -> str:
        """返回该算子类型在 Netron 中对应的类别颜色。"""
        op = op_type or ''
        category = self.OP_CATEGORY.get(op, '')
        return self.CATEGORY_COLORS.get(category, self.DEFAULT_COLOR)

    def _euc_to_color(self, euc: float | None) -> str:
        """将累计欧氏距离映射为绿(小)-黄(中)-红(大)颜色。

        映射：cos = 1.0 - t*0.4（t 为 euc 归一化值），使纯红区起点
        （cos<=0.8）正好落在 t=0.5——即欧氏距离达到本次范围一半才变红。
        """
        if euc is None:
            return '#b0b0b0'
        span = self.euc_color_max - self.euc_color_min
        t = 0.0 if span <= 0 else (euc - self.euc_color_min) / span
        return self._cos_to_color(1.0 - max(0.0, min(1.0, t)) * 0.4)

    @staticmethod
    def _text_color_for(bg: str) -> str:
        """根据背景色亮度选择文字颜色：浅背景用灰黑，深背景用白色。

        避免浅色类别（如 Input/Constant/Control 的 #eeeeee）上白色文字看不清。
        """
        bg = bg.lstrip('#')
        if len(bg) != 6:
            return '#fff'
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
        # 相对亮度 (ITU-R BT.709)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return '#333333' if lum > 140 else '#fff'

    def _layers(self) -> dict[str, int]:
        """dagre.js `rank()` 的 network-simplex 排名：longestPath + networkSimplex。

        1) longestPath：`rank(v)=max_dist - dist_to_sink(v)`，会把汇入同一个
           消费者（如 Concat）的所有分支输出对齐到消费者的前一层；
        2) networkSimplex（见 _network_simplex）：在最长路径初始排名上迭代
           换边改进，权衡"merge 输入深度 / 最终输出深度"，使 FPN 级联的多个
           SSH head 按深度对角错开，而非全部挤到同一层；
        3) 规范化：assignRankMinMax（平移使最小 rank=0）+ removeEmptyRanks
           （合并空层，使 rank 从 0 连续递增）。

        实现采用迭代逆拓扑（非递归，避免层深过深），环上节点兜底为 0。
        """
        from collections import deque

        layer_row = self.layer_row
        children = self.children

        # ---- Kahn 拓扑（源 -> 汇）----
        in_degree: dict[str, int] = {n: 0 for n in layer_row}
        for n in layer_row:
            for p in self.parents.get(n, []):
                if p in layer_row:
                    in_degree[n] += 1

        queue = deque(n for n in layer_row if in_degree[n] == 0)
        topo: list[str] = []
        while queue:
            n = queue.popleft()
            topo.append(n)
            for c in children.get(n, []):
                if c in layer_row:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)

        # ---- dist_to_sink（逆拓扑：先算汇）→ longestPath rank ----
        dist: dict[str, int] = {}
        for n in reversed(topo):
            ch = [c for c in children.get(n, []) if c in layer_row]
            if not ch:
                dist[n] = 0
            else:
                dist[n] = 1 + max(dist[c] for c in ch)
        for n in layer_row:  # 环上节点兜底 0
            if n not in dist:
                dist[n] = 0
        max_dist = max(dist.values(), default=0)
        rank = {n: max_dist - dist[n] for n in layer_row}

        # ---- networkSimplex：在最长路径初始排名上迭代改进 ----
        rank = self._network_simplex(rank)

        # ---- assignRankMinMax：平移使最小 rank = 0（dagre）----
        min_rank = min(rank.values(), default=0)
        rank = {n: r - min_rank for n, r in rank.items()}

        # ---- removeEmptyRanks：合并空 rank，使 rank 从 0 连续（dagre）----
        present_ranks = sorted(set(rank.values()))
        compact = {r: i for i, r in enumerate(present_ranks)}
        return {n: compact[r] for n, r in rank.items()}

    def _network_simplex(self, rank: dict[str, int]) -> dict[str, int]:
        """dagre.js `rank()` 的 networkSimplex：在最长路径初始排名上迭代改进。

        在 `feasibleTree`（紧树）基础上增加：
          - initLowLimValues：给树节点赋 low/lim/parent（DFS 区间，判断 ancestor）；
          - initCutValues：计算每条树边的割值（cutvalue）；
          - 循环 leaveEdge（cutvalue<0 的树边）→ enterEdge（跨割的最小平 slack
            非树边）→ exchangeEdges（换边并更新 rank），直到没有可换的负割边。

        相比 tight-tree，它会同时权衡"merge 输入深度 / 最终输出深度"，从而让 FPN
        级联的三个 SSH head 按深度对角错开，而不再是全挤到同一层。

        本图所有边 weight=1、minlen=1（简单图），故 simplify 为恒等。
        返回更新后的 rank dict（可能为负，调用方再 assignRankMinMax / removeEmptyRanks）。
        """
        from collections import defaultdict

        layer_row = self.layer_row
        children = self.children
        node_set = set(layer_row)
        nodes = list(layer_row)

        # 有向边 (v=父, w=子)，minlen=1, weight=1
        edge_list = []
        succs: dict[str, list[str]] = defaultdict(list)
        preds: dict[str, list[str]] = defaultdict(list)
        for p in layer_row:
            for c in children.get(p, []):
                if c in layer_row:
                    edge_list.append((p, c))
                    succs[p].append(c)
                    preds[c].append(p)
        edge_set = set(edge_list)

        def slack(e):
            v, w = e
            return rank[w] - rank[v] - 1

        # ---- feasibleTree：返回树边集合，并调整 rank ----
        tree_edges: set[frozenset] = set()
        in_tree: dict[str, bool] = {}
        start = next(iter(nodes))
        in_tree[start] = True
        while True:
            changed = True
            while changed:
                changed = False
                for (v, w) in edge_list:
                    if (v in in_tree) != (w in in_tree) and slack((v, w)) == 0:
                        nm = w if v in in_tree else v
                        if nm not in in_tree:
                            in_tree[nm] = True
                            tree_edges.add(frozenset((v, w)))
                            changed = True
            if len(in_tree) >= len(node_set):
                break
            min_key = float('inf')
            edge = None
            for (v, w) in edge_list:
                if (v in in_tree) != (w in in_tree):
                    key = slack((v, w))
                    if key < min_key:
                        min_key = key
                        edge = (v, w)
            if edge is None:
                break
            v, w = edge
            delta = slack((v, w)) if v in in_tree else -slack((v, w))
            for n in list(in_tree):
                rank[n] += delta

        # ---- initLowLimValues：DFS 给树节点赋 low/lim/parent ----
        def build_tree_adj():
            adj: dict[str, set[str]] = defaultdict(set)
            for e in tree_edges:
                a, b = tuple(e)
                adj[a].add(b)
                adj[b].add(a)
            return adj

        # tree_edges 是 set，必须按固定序扫描；否则字符串哈希随机化
        # （PYTHONHASHSEED）会让不同进程选到不同的负割边，布局结果不稳定。
        def tree_edges_sorted():
            return sorted(tree_edges, key=lambda e: tuple(sorted(e)))

        def init_low_lim():
            adj = build_tree_adj()
            if not adj:
                return {}, {}, {}, {}
            parent: dict[str, str | None] = {}
            pre: dict[str, int] = {}
            seen: set[str] = set()
            order_counter = 1
            # 每个连通块各取一个根，DFS 赋 pre/parent
            for root in sorted(adj.keys()):
                if root in seen:
                    continue
                stack = [(root, None, 0)]
                while stack:
                    v, p, state = stack.pop()
                    if state == 0:
                        if v in seen:
                            continue
                        seen.add(v)
                        parent[v] = p
                        pre[v] = order_counter
                        order_counter += 1
                        stack.append((v, p, 1))
                        for w in sorted(adj.get(v, ())):
                            if w not in seen:
                                stack.append((w, v, 0))
            # lim = 子树内最大 pre 下标
            lim: dict[str, int] = {}
            for v in sorted(seen, key=lambda x: -pre[x]):
                max_p = pre[v]
                for w in adj.get(v, ()):
                    if parent.get(w) == v:
                        max_p = max(max_p, lim.get(w, pre[w]))
                lim[v] = max_p
            low = {v: pre[v] for v in seen}
            return parent, low, lim, pre

        parent, low, lim, pre = init_low_lim()

        # ---- initCutValues：计算每条树边的割值 ----
        def init_cut_values():
            cutvalue: dict[frozenset, float] = {}
            adj = build_tree_adj()
            if not adj:
                return cutvalue
            # 叶子优先（按 pre 从大到小）——树边方向 child->parent
            edge_order = []
            for e in tree_edges_sorted():
                a, b = tuple(e)
                child = b if parent.get(b) == a else a
                par = a if child == b else b
                edge_order.append((e, child, par))
            edge_order.sort(key=lambda x: -pre[x[1]])
            for e, child, par in edge_order:
                cutvalue[e] = 1.0  # graphEdge weight（简单图 weight=1）
                childIsTail = (child, par) in edge_set
                incident = [(n, True) for n in succs.get(child, [])] + \
                           [(n, False) for n in preds.get(child, [])]
                for other, isOut in incident:
                    if other == par:
                        continue
                    pointsToHead = (isOut == childIsTail)
                    cutvalue[e] += 1.0 if pointsToHead else -1.0
                    te = frozenset((child, other))
                    if te in tree_edges:
                        otherCut = cutvalue.get(te, 1.0)
                        cutvalue[e] += -otherCut if pointsToHead else otherCut
            return cutvalue

        cutvalue = init_cut_values()

        # ---- leaveEdge：找 cutvalue<0 的树边 ----
        def leave_edge():
            for e in tree_edges_sorted():
                if cutvalue.get(e, 0) < 0:
                    return e
            return None

        def is_descendant(v_lowlim, root_lowlim):
            root_low, root_lim = root_lowlim
            _, v_lim = v_lowlim
            return root_low <= v_lim <= root_lim

        # ---- enterEdge：跨割且 slack 最小的非树边 ----
        def enter_edge(e):
            a, b = tuple(e)
            if (a, b) in edge_set:
                v, w = a, b
            else:
                v, w = b, a
            vLabel = (low.get(v, 0), lim.get(v, 0))
            wLabel = (low.get(w, 0), lim.get(w, 0))
            tailLabel = vLabel
            flip = False
            if vLabel[1] > wLabel[1]:
                tailLabel = wLabel
                flip = True
            min_key = float('inf')
            min_edge = None
            for (ev, ew) in edge_list:
                # 只允许换入"非树边"；否则 exchange 会变成 no-op 使树断开
                if frozenset((ev, ew)) in tree_edges:
                    continue
                if (flip == is_descendant((low.get(ev, 0), lim.get(ev, 0)), tailLabel)) and \
                   (flip != is_descendant((low.get(ew, 0), lim.get(ew, 0)), tailLabel)):
                    key = slack((ev, ew))
                    if key < min_key:
                        min_key = key
                        min_edge = (ev, ew)
            return min_edge

        def exchange_edges(e, f):
            nonlocal tree_edges, parent, low, lim, pre, cutvalue
            tree_edges.discard(e)
            tree_edges.add(frozenset(f))
            parent, low, lim, pre = init_low_lim()
            # 从根 BFS，按树边方向更新 rank（minlen=1）
            adj = build_tree_adj()
            if adj:
                root = next((n for n in nodes if parent.get(n) is None), nodes[0])
                stack = [root]
                seen = {root}
                while stack:
                    v = stack.pop()
                    for w in sorted(adj.get(v, ())):
                        if w in seen:
                            continue
                        seen.add(w)
                        if (v, w) in edge_set:
                            rank[w] = rank[v] + 1
                        elif (w, v) in edge_set:
                            rank[w] = rank[v] - 1
                        else:
                            rank[w] = rank[v]
                        stack.append(w)
            cutvalue = init_cut_values()

        # ---- 主循环 ----
        guard = 0
        while guard < 1000:
            e = leave_edge()
            if e is None:
                break
            f = enter_edge(e)
            if f is None:
                break
            exchange_edges(e, f)
            guard += 1

        return rank

    def _compute_layout(
        self, depth: dict[str, int]
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]],
               dict[tuple[str, str], list[str]], dict[int, int]]:
        """dagre 风格 Sugiyama 布局（Netron 使用的布局算法，保持 TB 纵向）。

        流水线：
          1. normalize —— 把跨多层长边拆成虚拟节点（每个中间层插一个），
             使所有边只跨一层；虚拟节点与真实节点共用同一套排序/定位，
             不再"绕到图外侧"。
          2. order —— 重心法(barycenter) + 交叉计数反馈 + transpose 交换，
             最小化边交叉。
          3. position —— Brandes-Köpf 坐标分配（中位数邻居对齐成"块" +
             块图横向压缩，4 个方向各算一遍取最窄再平衡），让直线链保持
             同一列、只有分叉才横向错开。

        Returns:
            tuple[real_pos, dummy_pos, edge_routes, level_sizes]:
              real_pos: 真实节点 name -> (x, y)
              dummy_pos: 虚拟节点 id -> (x, y)
              edge_routes: (a, b) -> 该边经过的虚拟节点 id 列表
              level_sizes: 层级 -> 节点数
        """
        from collections import defaultdict

        node_w = 240.0
        node_h = 86.0
        node_sep = self.node_sep
        edge_sep = self.edge_sep
        rank_sep = self.rank_sep

        # 用确定性的层名序作种子，避免布局依赖框架提供的节点顺序
        # （RKNN/QNN 因输入行序不同而互为镜像）。rank 仍由 _layers() 计算，
        # 这里只规范化同 rank 内的相对顺序，供 init_order 与后继遍历作为
        # 确定性的平局打破。
        ordered_real = sorted(self.layer_row.keys(), reverse=True)
        real = ordered_real
        real_set = set(ordered_real)
        seq_of = {n: i for i, n in enumerate(ordered_real)}

        # ---- 1. normalize：构建增强图（真实节点 + 虚拟节点） ----
        nodes: dict[str, dict] = {}
        for n in ordered_real:
            nodes[n] = {
                'rank': depth.get(n, 0),
                'width': node_w,
                'height': node_h,
                'dummy': False,
                'seq': seq_of[n],
            }

        edge_routes: dict[tuple[str, str], list[str]] = {}
        aug_in: dict[str, list[str]] = defaultdict(list)
        aug_out: dict[str, list[str]] = defaultdict(list)
        dummy_counter = [len(real)]

        def make_dummy(rank: int) -> str:
            dummy_counter[0] += 1
            did = f'__dummy__{dummy_counter[0]}'
            nodes[did] = {'rank': rank, 'width': 0.0, 'height': 0.0,
                          'dummy': True, 'seq': dummy_counter[0]}
            return did

        def add_edge(u: str, v: str) -> None:
            aug_out[u].append(v)
            aug_in[v].append(u)

        for a, blist in self.children.items():
            if a not in real_set:
                continue
            for b in blist:
                if b not in real_set:
                    continue
                da = nodes[a]['rank']
                db = nodes[b]['rank']
                if db <= da + 1:
                    add_edge(a, b)
                    edge_routes[(a, b)] = []
                else:
                    chain: list[str] = []
                    prev = a
                    for r in range(da + 1, db):
                        d = make_dummy(r)
                        chain.append(d)
                        add_edge(prev, d)
                        prev = d
                    add_edge(prev, b)
                    edge_routes[(a, b)] = chain

        # 规范化后继/前驱遍历顺序（按 seq），使 init_order 的 DFS 遍历与层间
        # 约束都与框架输入顺序无关，从而消除 RKNN/QNN 的镜像差异。
        for v in aug_out:
            aug_out[v].sort(key=lambda u: nodes[u].get('seq', 0))
        for v in aug_in:
            aug_in[v].sort(key=lambda u: nodes[u].get('seq', 0))

        # ---- 2. order：交叉最小化 ----
        layering = self._dagre_order(nodes, aug_in, aug_out)

        # ---- 3. position：Brandes-Köpf ----
        x_pos, y_pos = self._dagre_position(
            layering, nodes, aug_in, aug_out, node_sep, edge_sep, rank_sep,
        )

        # 收尾：把每条长边的虚拟节点链夹在"源-目标 x 走廊"内，避免残差/跳连
        # 被顶到远处的高速公路列（如 unisal /cnn/Slice_5 的残差被推到 x=1510，
        # 而源/目标都在 x=765）。只收紧过大的水平摆幅，不改变真实节点位置，
        # 也不影响 order/rank（故与镜像/确定性修复无关）。
        #
        # 允许的水平摆幅按边跨度（dummy 数）放宽：短边（如残差）仍收紧在
        # 源-目标附近；横贯模型的长边则允许绕远，避免被硬性拉回同一列。
        pad = node_sep * 1.5
        for (a, b), route in edge_routes.items():
            if not route:
                continue
            allow = pad + len(route) * edge_sep
            lo = min(x_pos[a], x_pos[b]) - allow
            hi = max(x_pos[a], x_pos[b]) + allow
            for d in route:
                x_pos[d] = min(max(x_pos[d], lo), hi)

        real_pos = {n: (x_pos[n], y_pos[n]) for n in real}
        dummy_pos = {n: (x_pos[n], y_pos[n]) for n in nodes if nodes[n]['dummy']}
        level_sizes = {r: len(layer) for r, layer in enumerate(layering)}

        return real_pos, dummy_pos, edge_routes, level_sizes

    def _dagre_order(
        self,
        nodes: dict[str, dict],
        aug_in: dict[str, list[str]],
        aug_out: dict[str, list[str]],
    ) -> list[list[str]]:
        """dagre.js `order()` 的忠实移植（非 compound 图）。

        与旧的重心扫描不同，这里对齐 dagre 的做法：
          - 每个层级边界用"层图"排序（down 用前驱、up 用后继的 barycenter）；
          - `resolveConflicts`：用约束图把会违反已定顺序的节点合并；
          - 交替 down/up 多轮扫描 + transpose 收尾，最小化边交叉。
        这是 SSH 头部三条并行分支（3×3 / 5×5 / 7×7）能整齐排列、不互相交叉的关键。

        Returns:
            layering: list[list[str]]（按 rank/order 排列）。
        """
        from functools import cmp_to_key

        max_rank = max(nd['rank'] for nd in nodes.values())
        if max_rank < 0:
            return []

        def set_order(v: str, o: int) -> None:
            nodes[v]['order'] = o

        def neighbors_rel(v: str, relationship: bool) -> list[str]:
            # relationship=True（down）用前驱；False（up）用后继
            return aug_in.get(v, []) if relationship else aug_out.get(v, [])

        # ---- initOrder（dagre：按 rank 升序遍历起点，对每个起点做后继优先 DFS，
        #      节点首次被访问时按 rank 归入对应层）----
        def init_order() -> list[list[str]]:
            visited: set[str] = set()
            layers: list[list[str]] = [[] for _ in range(max_rank + 1)]
            ordered_vs = sorted(nodes.keys(),
                                 key=lambda n: (nodes[n]['rank'], nodes[n].get('seq', 0)))
            for start in ordered_vs:
                if start in visited:
                    continue
                stack = [start]
                while stack:
                    v = stack.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    layers[nodes[v]['rank']].append(v)
                    # 逆序压栈，使出栈顺序 = aug_out 后继顺序（等价官方递归 dfs）
                    for w in reversed(aug_out.get(v, [])):
                        if w not in visited:
                            stack.append(w)
            return layers

        layering = init_order()

        def assign_order(layers: list[list[str]]) -> None:
            for layer in layers:
                for i, v in enumerate(layer):
                    set_order(v, i)

        assign_order(layering)

        # ---- crossCount（本图所有边 weight=1 → 等价于逆序对数）----
        def cross_count(layers: list[list[str]]) -> int:
            total = 0
            for r in range(1, len(layers)):
                south_pos = {v: i for i, v in enumerate(layers[r])}
                entries: list[int] = []
                for v in layers[r - 1]:
                    ws = [south_pos[w] for w in aug_out.get(v, []) if w in south_pos]
                    ws.sort()
                    entries.extend(ws)
                total += self._count_inversions(entries, len(layers[r]))
            return total

        # ---- barycenter ----
        def barycenter(v: str, relationship: bool) -> dict:
            nb = neighbors_rel(v, relationship)
            s = 0.0
            w = 0.0
            for u in nb:
                o = nodes.get(u, {}).get('order')
                if o is None:
                    continue
                s += o
                w += 1.0
            if w == 0:
                return {'v': v}
            return {'v': v, 'barycenter': s / w, 'weight': w}

        # ---- resolveConflicts（约束图冲突合并，改写自 dagre.js）----
        def resolve_conflicts(entries: list[dict], cg: dict[str, list[str]]) -> list[dict]:
            mapped: dict[str, dict] = {}
            for i, entry in enumerate(entries):
                tmp = {'indegree': 0, 'in': [], 'out': [], 'vs': [entry['v']], 'i': i}
                if 'barycenter' in entry:
                    tmp['barycenter'] = entry['barycenter']
                    tmp['weight'] = entry['weight']
                mapped[entry['v']] = tmp
            for frm, tos in cg.items():
                ev = mapped.get(frm)
                if not ev:
                    continue
                for to in tos:
                    ew = mapped.get(to)
                    if ew:
                        ew['indegree'] += 1
                        ev['out'].append(ew)
            source_set = [e for e in mapped.values() if e['indegree'] == 0]
            results: list[dict] = []

            def handle_in(v_entry: dict):
                def fn(u_entry: dict) -> None:
                    if u_entry.get('merged'):
                        return
                    if (u_entry.get('barycenter') is None
                            or v_entry.get('barycenter') is None
                            or u_entry.get('barycenter', 0) >= v_entry.get('barycenter', 0)):
                        s = 0.0
                        w = 0.0
                        if v_entry.get('weight'):
                            s += v_entry['barycenter'] * v_entry['weight']
                            w += v_entry['weight']
                        if u_entry.get('weight'):
                            s += u_entry['barycenter'] * u_entry['weight']
                            w += u_entry['weight']
                        v_entry['vs'] = u_entry['vs'] + v_entry['vs']
                        if w:
                            v_entry['barycenter'] = s / w
                        else:
                            v_entry.pop('barycenter', None)
                        v_entry['weight'] = w
                        v_entry['i'] = min(u_entry['i'], v_entry['i'])
                        u_entry['merged'] = True
                return fn

            def handle_out(v_entry: dict):
                def fn(w_entry: dict) -> None:
                    w_entry['in'].append(v_entry)
                    w_entry['indegree'] -= 1
                    if w_entry['indegree'] == 0:
                        source_set.append(w_entry)
                return fn

            while source_set:
                entry = source_set.pop()
                results.append(entry)
                for u in reversed(entry['in']):
                    handle_in(entry)(u)
                for w in entry['out']:
                    handle_out(entry)(w)

            out: list[dict] = []
            for e in results:
                if e.get('merged'):
                    continue
                value = {'vs': e['vs'], 'i': e['i']}
                if 'barycenter' in e:
                    value['barycenter'] = e['barycenter']
                    value['weight'] = e['weight']
                out.append(value)
            return out

        # ---- sort（带 bias 与 unsortable 吸收，改写自 dagre.js）----
        def sort(entries: list[dict], bias_right: bool) -> dict:
            lhs = [e for e in entries if 'barycenter' in e]
            rhs = [e for e in entries if 'barycenter' not in e]
            unsortable = sorted(rhs, key=lambda e: -e['i'])

            def compare(a: dict, b: dict) -> int:
                if a['barycenter'] < b['barycenter']:
                    return -1
                if a['barycenter'] > b['barycenter']:
                    return 1
                return (b['i'] - a['i']) if bias_right else (a['i'] - b['i'])

            sortable = sorted(lhs, key=cmp_to_key(compare))

            def consume_unsortable(vs: list, un: list, index: int) -> int:
                while un and un[-1]['i'] <= index:
                    last = un.pop()
                    vs.append(last['vs'])
                    index += 1
                return index

            vs: list = []
            s = 0.0
            w = 0.0
            vs_index = consume_unsortable(vs, unsortable, 0)
            for entry in sortable:
                vs_index += len(entry['vs'])
                vs.append(entry['vs'])
                s += entry['barycenter'] * entry['weight']
                w += entry['weight']
                vs_index = consume_unsortable(vs, unsortable, vs_index)
            result = {'vs': [item for sub in vs for item in sub]}
            if w:
                result['barycenter'] = s / w
                result['weight'] = w
            return result

        # ---- sortSubgraph（非 compound：movable = rank 内全部真实+虚拟节点）----
        def sort_rank(rank_nodes: list[str], cg: dict[str, list[str]], bias_right: bool,
                      relationship: bool) -> list[str]:
            entries = [barycenter(v, relationship) for v in rank_nodes]
            entries = resolve_conflicts(entries, cg)
            return sort(entries, bias_right)['vs']

        # ---- sweepLayerGraphs：对一组 rank 边界做一次扫描，并记录层内约束 ----
        def sweep_ranks(ranks_to_sweep: list[int], bias_right: bool, relationship: bool) -> None:
            cg: dict[str, list[str]] = {}
            for r in ranks_to_sweep:
                vs = sort_rank(list(layering[r]), cg, bias_right, relationship)
                layering[r] = vs
                for i, v in enumerate(vs):
                    set_order(v, i)
                for i in range(len(vs) - 1):
                    cg.setdefault(vs[i], []).append(vs[i + 1])

        down_ranks = list(range(1, max_rank + 1))
        up_ranks = list(range(max_rank - 1, -1, -1))

        best = [layer[:] for layer in layering]
        best_cc = cross_count(best)
        i = 0
        last_best = 0
        while last_best < 4 and i < 48:
            bias_right = (i % 4) >= 2
            if i % 2 == 0:
                sweep_ranks(up_ranks, bias_right, relationship=False)
            else:
                sweep_ranks(down_ranks, bias_right, relationship=True)
            cc = cross_count(layering)
            if cc < best_cc:
                best_cc = cc
                best = [layer[:] for layer in layering]
                last_best = 0
            else:
                last_best += 1
            i += 1

        layering = self._dagre_transpose(best, aug_in, aug_out)
        assign_order(layering)
        return layering

    @staticmethod
    def _count_inversions(entries: list[int], size: int) -> int:
        """统计 entries（元素取值 [0, size)）中的逆序对数量（Fenwick 树）。"""
        tree = [0] * (size + 1)

        def add(i: int) -> None:
            i += 1
            while i <= size:
                tree[i] += 1
                i += i & -i

        def prefix(i: int) -> int:
            s = 0
            while i > 0:
                s += tree[i]
                i -= i & -i
            return s

        inv = 0
        seen = 0
        for p in entries:
            inv += seen - prefix(p + 1)
            add(p)
            seen += 1
        return inv

    def _dagre_transpose(
        self,
        layering: list[list[str]],
        aug_in: dict[str, list[str]],
        aug_out: dict[str, list[str]],
    ) -> list[list[str]]:
        """局部 transpose：交换同一层相邻节点，若减少交叉则保留（类 dagre）。"""

        def cross_between(north: list[str], south: list[str]) -> int:
            south_pos = {v: j for j, v in enumerate(south)}
            entries = []
            for v in north:
                for w in aug_out.get(v, []):
                    if w in south_pos:
                        entries.append(south_pos[w])
            return self._count_inversions(entries, len(south))

        def local_cross(r: int) -> int:
            total = 0
            if r > 0:
                total += cross_between(layering[r - 1], layering[r])
            if r < len(layering) - 1:
                total += cross_between(layering[r], layering[r + 1])
            return total

        for r in range(len(layering)):
            layer = layering[r]
            improved = True
            while improved:
                improved = False
                for j in range(len(layer) - 1):
                    before = local_cross(r)
                    layer[j], layer[j + 1] = layer[j + 1], layer[j]
                    after = local_cross(r)
                    if after < before:
                        improved = True
                    else:
                        layer[j], layer[j + 1] = layer[j + 1], layer[j]
        return layering

    def _dagre_position(
        self,
        layering: list[list[str]],
        nodes: dict[str, dict],
        aug_in: dict[str, list[str]],
        aug_out: dict[str, list[str]],
        node_sep: float,
        edge_sep: float,
        rank_sep: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Brandes-Köpf 坐标分配：返回 (x_pos, y_pos)。"""
        from collections import defaultdict, deque
        import math

        for r, layer in enumerate(layering):
            for j, v in enumerate(layer):
                nodes[v]['order'] = j

        y_pos: dict[str, float] = {}
        y = 0.0
        for layer in layering:
            max_h = max((nodes[v]['height'] for v in layer), default=86.0)
            for v in layer:
                y_pos[v] = y + max_h / 2.0
            y += max_h + rank_sep

        conflicts = self._find_type1_conflicts(layering, nodes, aug_in)
        # 合并 type-2 冲突：两条相互交叉的长边（虚边）不允许被对齐到同一列
        for v, ws in self._find_type2_conflicts(layering, nodes, aug_in).items():
            conflicts.setdefault(v, set()).update(ws)

        def has_conflict(v: str, w: str) -> bool:
            if v > w:
                v, w = w, v
            return w in conflicts.get(v, ())

        xss: dict[str, dict[str, float]] = {}
        for vertical in ('u', 'd'):
            adj = layering if vertical == 'u' else list(reversed(layering))
            for horizontal in ('l', 'r'):
                adj2 = [list(reversed(layer)) for layer in adj] if horizontal == 'r' else adj
                neighbor = aug_in if vertical == 'u' else aug_out
                root, align = self._vertical_alignment(adj2, nodes, neighbor, has_conflict)
                xs = self._horizontal_compaction(adj2, root, align, nodes, node_sep, edge_sep)
                if horizontal == 'r':
                    xs = {v: -x for v, x in xs.items()}
                xss[vertical + horizontal] = xs

        def width_of(xs: dict[str, float]) -> float:
            lo = min(x - nodes[v]['width'] / 2.0 for v, x in xs.items())
            hi = max(x + nodes[v]['width'] / 2.0 for v, x in xs.items())
            return hi - lo

        min_xs = min(xss.values(), key=width_of)
        align_range = (min(min_xs.values()), max(min_xs.values()))
        for vertical in ('u', 'd'):
            for horizontal in ('l', 'r'):
                key = vertical + horizontal
                xs = xss[key]
                if xs is not min_xs:
                    rng = (min(xs.values()), max(xs.values()))
                    delta = (align_range[0] - rng[0]) if horizontal == 'l' else (align_range[1] - rng[1])
                    if delta:
                        xss[key] = {v: x + delta for v, x in xs.items()}

        x_pos: dict[str, float] = {}
        for v in (v for layer in layering for v in layer):
            vals = sorted([xss['ul'][v], xss['ur'][v], xss['dl'][v], xss['dr'][v]])
            x_pos[v] = (vals[1] + vals[2]) / 2.0
        return x_pos, y_pos

    @staticmethod
    def _vertical_alignment(
        layering: list[list[str]],
        nodes: dict[str, dict],
        neighbor: dict[str, list[str]],
        has_conflict,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """中位数邻居垂直对齐，返回 (root, align)。"""
        import math

        root: dict[str, str] = {}
        align: dict[str, str] = {}
        pos: dict[str, int] = {}
        for layer in layering:
            for j, v in enumerate(layer):
                root[v] = v
                align[v] = v
                pos[v] = j

        for layer in layering:
            prev_idx = -1
            for v in layer:
                ws = [w for w in neighbor.get(v, []) if w in pos]
                if not ws:
                    continue
                ws.sort(key=lambda w: pos[w])
                mp = (len(ws) - 1) / 2.0
                for i in range(math.floor(mp), math.ceil(mp) + 1):
                    w = ws[i]
                    if align[v] == v and prev_idx < pos[w] and not has_conflict(v, w):
                        x = root[w]
                        align[w] = v
                        align[v] = x
                        root[v] = x
                        prev_idx = pos[w]
        return root, align

    @staticmethod
    def _horizontal_compaction(
        layering: list[list[str]],
        root: dict[str, str],
        align: dict[str, str],
        nodes: dict[str, dict],
        node_sep: float,
        edge_sep: float,
    ) -> dict[str, float]:
        """块图横向压缩，返回每个节点的 x（= 其所在块的 x）。"""
        from collections import defaultdict, deque

        sep_map: dict[tuple[str, str], float] = {}
        block_set: set[str] = set()
        for layer in layering:
            prev = None
            for v in layer:
                vroot = root[v]
                block_set.add(vroot)
                if prev is not None:
                    uroot = root[prev]
                    un = nodes[prev]
                    vn = nodes[v]
                    sep = (vn['width'] / 2.0
                           + (edge_sep if vn['dummy'] else node_sep) / 2.0
                           + (edge_sep if un['dummy'] else node_sep) / 2.0
                           + un['width'] / 2.0)
                    key = (uroot, vroot)
                    if key not in sep_map or sep > sep_map[key]:
                        sep_map[key] = sep
                prev = v

        block_out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        block_in: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (u, v), sep in sep_map.items():
            block_out[u].append((v, sep))
            block_in[v].append((u, sep))

        indeg = {b: len(block_in[b]) for b in block_set}
        queue = deque(sorted(b for b in block_set if indeg[b] == 0))
        topo: list[str] = []
        while queue:
            b = queue.popleft()
            topo.append(b)
            for (c, _s) in block_out[b]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        for b in block_set:  # 环兜底
            if b not in topo:
                topo.append(b)

        xs: dict[str, float] = {b: 0.0 for b in block_set}
        for b in topo:
            for (c, s) in block_out[b]:
                xs[c] = max(xs[c], xs[b] + s)
        for b in reversed(topo):
            # 第二遍（对齐 dagre pass2）：对所有块根（含 dummy 块）取
            # max(自身, 最小出边坐标-间距)，把块向右推以消除右侧空白。
            best = min((xs[c] - s for c, s in block_out[b]), default=None)
            if best is not None:
                xs[b] = max(xs[b], best)

        result: dict[str, float] = {}
        for layer in layering:
            for v in layer:
                result[v] = xs[root[v]]
        return result

    @staticmethod
    def _find_type1_conflicts(
        layering: list[list[str]],
        nodes: dict[str, dict],
        aug_in: dict[str, list[str]],
    ) -> dict[str, set[str]]:
        """找出 type-1 冲突（非内部段穿过内部段的节点对），用于 BK 对齐。"""
        conflicts: dict[str, set[str]] = {}

        def add_conflict(v: str, w: str) -> None:
            if v > w:
                v, w = w, v
            conflicts.setdefault(v, set()).add(w)

        if not layering:
            return conflicts
        prev = layering[0]
        for k in range(1, len(layering)):
            layer = layering[k]
            k0 = 0
            scan_pos = 0
            prev_len = len(prev)
            last_node = layer[-1] if layer else None
            for i, v in enumerate(layer):
                w = None
                if nodes[v]['dummy']:
                    for u in aug_in.get(v, []):
                        if nodes[u]['dummy']:
                            w = u
                            break
                if w is not None or v == last_node:
                    k1 = nodes[w]['order'] if w is not None else prev_len
                    for scan_node in layer[scan_pos:i + 1]:
                        for u in aug_in.get(scan_node, []):
                            u_order = nodes[u]['order']
                            if (u_order < k0 or k1 < u_order) and not (
                                nodes[u]['dummy'] and nodes[scan_node]['dummy']
                            ):
                                add_conflict(u, scan_node)
                    scan_pos = i + 1
                    k0 = k1
            prev = layer
        return conflicts

    @staticmethod
    def _find_type2_conflicts(
        layering: list[list[str]],
        nodes: dict[str, dict],
        aug_in: dict[str, list[str]],
    ) -> dict[str, set[str]]:
        """找出 type-2 冲突（两条虚边在相邻层的内段互相交叉），用于 BK 对齐。

        改写自 dagre.js `findType2Conflicts`（非 compound 分支）：扫描相邻两层
        north->south，当 south 层的虚节点（长边中间节点）v 的虚前驱 u 落在
        "上一个内段边界 ~ 当前内段边界"之外时，标记 (u, v) 冲突，避免两条
        相互交叉的长边被垂直对齐到同一列而重叠。
        """
        conflicts: dict[str, set[str]] = {}

        def add_conflict(v: str, w: str) -> None:
            if v > w:
                v, w = w, v
            conflicts.setdefault(v, set()).add(w)

        def scan(south, south_pos, south_end, prev_north_border, next_north_border):
            for i in range(south_pos, south_end):
                v = south[i]
                if nodes[v]['dummy']:
                    for u in aug_in.get(v, []):
                        if nodes[u]['dummy']:
                            u_order = nodes[u]['order']
                            if u_order < prev_north_border or u_order > next_north_border:
                                add_conflict(u, v)

        for i in range(len(layering) - 1):
            north = layering[i]
            south = layering[i + 1]
            prev_north_pos = -1
            next_north_pos = -1
            south_pos = 0
            for south_lookahead, v in enumerate(south):
                if nodes[v]['dummy']:
                    preds = aug_in.get(v, [])
                    if preds:
                        next_north_pos = nodes[preds[0]]['order']
                        scan(south, south_pos, south_lookahead,
                             prev_north_pos, next_north_pos)
                        south_pos = south_lookahead
                        prev_north_pos = next_north_pos
            scan(south, south_pos, len(south), next_north_pos, len(north))
        return conflicts

    @staticmethod
    def _intersect_rect(
        node_center: tuple[float, float],
        point: tuple[float, float],
        node_w: float,
        node_h: float,
    ) -> tuple[float, float]:
        """计算从节点中心指向 point 的射线与节点矩形的交点（dagre intersectRect）。"""
        x, y = node_center
        px, py = point
        dx = px - x
        dy = py - y
        if dx == 0.0 and dy == 0.0:
            return (x, y + node_h / 2.0)
        w = node_w / 2.0
        h = node_h / 2.0
        if abs(dy) * w > abs(dx) * h:
            if dy < 0:
                h = -h
            return (x + h * dx / dy, y + h)
        if dx < 0:
            w = -w
        return (x + w, y + w * dy / dx)

    # ------------------------------------------------------------------
    # 构建与渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt5(x: float | None) -> str:
        """5 位有效数字显示：整数部分位数决定小数位数（左右共享共 5 位）。

        例：100.00（3+2）、1.0000（1+4）、0.9998（0+4，前导 0 占一位）、16.237（2+3）。
        """
        if x is None:
            return '-'
        n = len(str(int(abs(x))))
        p = max(0, 5 - n)
        return f'{x:.{p}f}'

    def _svg_node(self, name: str, op: str, row: dict) -> tuple[str, str]:
        """
        生成 Netron 风格节点的内联 SVG 矢量图 data URI。

        上半栏：算子类型（背景色 = 该算子代表色，白字）
        下半栏：精度数据（背景色 = 累积余弦误差 entire_cos 颜色）

        Returns:
            tuple[str, str]: (svg data_uri, tooltip 纯文本)
        """
        import base64

        entire_cos = row.get('entire_cos')
        entire_euc = row.get('entire_euc')
        single_cos = row.get('single_cos')
        single_euc = row.get('single_euc')

        # 上半栏：算子类型色
        top_fill = self._op_color(op)
        # 上半栏文字色：随背景亮度自适应（浅背景用灰黑，深背景用白）
        top_text = self._text_color_for(top_fill)
        # 下半栏左右两块：左侧累计余弦精度，右侧累计欧氏距离精度
        cos_fill = self._cos_to_color(entire_cos)
        euc_fill = self._euc_to_color(entire_euc)

        # 精度文本（4 个值全部显示，5 位有效数字）
        e_cos = self._fmt5(entire_cos)
        e_euc = self._fmt5(entire_euc)
        s_cos = self._fmt5(single_cos)
        s_euc = self._fmt5(single_euc)

        # 尺寸：宽 240，类型栏高 34，精度栏高 52
        w, top_h, bot_h = 240, 34, 52
        h = top_h + bot_h
        r = 12  # 圆角半径

        def esc(t: str) -> str:
            return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # 用 clipPath 实现整体圆角，上下两半各占一个半圆角矩形
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<defs><clipPath id="cp">'
            f'<rect x="0" y="0" width="{w}" height="{h}" rx="{r}" ry="{r}"/>'
            f'</clipPath>'
            f'<linearGradient id="accuracy-gradient" x1="0%" y1="0%" x2="100%" y2="0%">'
            f'<stop offset="0%" stop-color="{cos_fill}"/>'
            f'<stop offset="37.5%" stop-color="{cos_fill}"/>'
            f'<stop offset="62.5%" stop-color="{euc_fill}"/>'
            f'<stop offset="100%" stop-color="{euc_fill}"/>'
            f'</linearGradient></defs>'
            # 上半栏：算子类型色（仅顶部两角圆角）
            f'<rect x="0" y="0" width="{w}" height="{top_h}" fill="{top_fill}" clip-path="url(#cp)"/>'
            f'<text x="{w/2}" y="{top_h-10}" font-family="Arial" font-size="16" fill="{top_text}" '
            f'text-anchor="middle" font-weight="bold">{esc(op)}</text>'
            # 下半栏：左端为累计余弦颜色，向右平滑过渡到累计欧氏距离颜色
            f'<rect x="0" y="{top_h}" width="{w}" height="{bot_h}" '
            f'fill="url(#accuracy-gradient)" clip-path="url(#cp)"/>'
            f'<text x="8" y="{top_h+18}" font-family="monospace" font-size="12" fill="#000">'
            f'<tspan font-weight="bold">single:</tspan> cos={esc(s_cos)} euc={esc(s_euc)}</text>'
            f'<text x="8" y="{top_h+38}" font-family="monospace" font-size="12" fill="#000">'
            f'<tspan font-weight="bold">entire:</tspan> cos={esc(e_cos)} euc={esc(e_euc)}</text>'
            # 外框描边（圆角）
            f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="{r}" ry="{r}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
            f'</svg>'
        )
        b64 = base64.b64encode(svg.encode()).decode('ascii')
        data_uri = 'data:image/svg+xml;base64,' + b64

        # tooltip 纯文本：真实节点名 + 详细精度
        title_lines = [name, f'op_type: {op}']
        if entire_cos is not None:
            title_lines.append(f'entire_cos={entire_cos:.6f}  euc={entire_euc}')
        else:
            title_lines.append('entire_cos=n/a')
        if single_cos is not None:
            title_lines.append(f'single_cos={single_cos:.6f}  euc={single_euc}')
        else:
            title_lines.append('single_cos=n/a')
        return data_uri, '\n'.join(title_lines)

    @staticmethod
    def _netron_curve_path(points: list[tuple[float, float]]) -> str:
        """复刻 Netron `grapher.Edge.Curve`（source/grapher.js）的漂浮式三次贝塞尔路径。

        与"精确过每个点"的 Catmull-Rom 不同，它用当前的 (x0,y0)/(x1,y1) 与下一个
        点做加权得到控制点，使曲线在点列附近"漂浮"：转角更圆润，不会在折点处
        外凸过冲。适合长边沿垂直走廊"拐出-直线-拐回"的绕行。

        本实现是 Netron 源码的逐行移植：
          - point() 状态机：0→moveTo, 1→(仅推进状态), 2→lineTo(前导点)+curve,
            3→curve；其中 curve() 只写贝塞尔段、不推进 (x0,y0)/(x1,y1)，
            推进统一在 point 末尾完成（旧实现把推进写进 curve 导致 x0==x1 退化）。
          - 每个点处理完后，若为末点则按末态补尾：state==3 时 curve(x1,y1)+lineTo(x1,y1)，
            state==2 时仅 lineTo(x1,y1)，使终点精确落在最后一个点上。
        """
        pts = [(float(x), float(y)) for x, y in points]
        n = len(pts)
        if n < 2:
            return ''

        data = ''
        x0 = y0 = x1 = y1 = float('nan')
        state = 0

        def curve(x: float, y: float) -> None:
            nonlocal data
            # 与 Netron 一致：curve 不推进 x0/x1/y0/y1，避免后续段退化
            data += (
                f'C{(2 * x0 + x1) / 3:.1f},{(2 * y0 + y1) / 3:.1f},'
                f'{(x0 + 2 * x1) / 3:.1f},{(y0 + 2 * y1) / 3:.1f},'
                f'{(x0 + 4 * x1 + x) / 6:.1f},{(y0 + 4 * y1 + y) / 6:.1f}'
            )

        for i, (x, y) in enumerate(pts):
            if state == 0:
                state = 1
                data += f'M{x:.1f},{y:.1f}'
            elif state == 1:
                state = 2
            elif state == 2:
                state = 3
                # Netron 的前导 line_to：从起点迈向 (5*x0+x1)/6，
                # 为第一条贝塞尔提供明确的起始切线
                data += f'L{(5 * x0 + x1) / 6:.1f},{(5 * y0 + y1) / 6:.1f}'
                curve(x, y)
            else:
                curve(x, y)

            x0, x1 = x1, x
            y0, y1 = y1, y

            if i == n - 1:
                if state == 3:
                    curve(x1, y1)
                    data += f'L{x1:.1f},{y1:.1f}'
                elif state == 2:
                    data += f'L{x1:.1f},{y1:.1f}'
        return data

    def _build_edge_paths(
        self,
        real_pos: dict[str, tuple[float, float]],
        dummy_pos: dict[str, tuple[float, float]],
        edge_routes: dict[tuple[str, str], list[str]],
    ) -> list[tuple[list[tuple[float, float]], str, str]]:
        """为每条真实边构建完整路径点列（Netron 风格）。

        Returns:
            list[(points, color, title)]:
              points: [源边界交点] + 虚拟节点点列 + [目标边界交点]
              color: 边颜色（源节点累积精度色）
              title: tooltip 文本
        """
        node_w = 240.0
        node_h = 86.0
        paths: list[tuple[list[tuple[float, float]], str, str]] = []

        for a, blist in self.children.items():
            if a not in real_pos:
                continue
            for tb in blist:
                if tb not in real_pos:
                    continue

                src_ec = self.layer_row[a].get('entire_cos')
                ec_color = self._cos_to_color(src_ec)
                dst_ec = self.layer_row[tb].get('entire_cos')
                title = (
                    f'{a} -> {tb}\nentire_cos={dst_ec:.6f}'
                    if dst_ec is not None else f'{a} -> {tb}'
                )

                route = edge_routes.get((a, tb), [])
                if route:
                    # 长边：源边界 → 各虚拟节点 → 目标边界（曲线沿层间间隙行进）
                    first = dummy_pos[route[0]]
                    last = dummy_pos[route[-1]]
                    src_pt = self._intersect_rect(real_pos[a], first, node_w, node_h)
                    dst_pt = self._intersect_rect(real_pos[tb], last, node_w, node_h)
                    pts = [src_pt] + [dummy_pos[d] for d in route] + [dst_pt]
                else:
                    # 短边（相邻层）：边界交点 → 边界交点
                    src_pt = self._intersect_rect(real_pos[a], real_pos[tb], node_w, node_h)
                    dst_pt = self._intersect_rect(real_pos[tb], real_pos[a], node_w, node_h)
                    col_dx = real_pos[tb][0] - real_pos[a][0]
                    span = dst_pt[0] - src_pt[0]
                    if abs(col_dx) < 1.0:
                        # 同列：两点直线
                        pts = [src_pt, dst_pt]
                    else:
                        # 并排/分叉：外凸贝塞尔曲线，方向朝扇出更多的一侧
                        out_n = len(self.children.get(a, []))
                        in_n = len(self.parents.get(tb, []))
                        sgn = 1.0 if span > 0 else -1.0
                        if in_n > out_n:
                            sgn = -sgn
                        mx = (src_pt[0] + dst_pt[0]) / 2.0 + sgn * 0.3 * abs(span)
                        my = (src_pt[1] + dst_pt[1]) / 2.0
                        pts = [src_pt, (mx, my), dst_pt]
                paths.append((pts, ec_color, title))
        return paths

    def render(self, show: bool = True) -> str:
        """渲染网络图为自包含 SVG HTML（无外部依赖）。

        边用 Netron 风格漂浮贝塞尔曲线（_netron_curve_path）穿过所有虚拟节点点列，
        实现真正平滑的绕行长弧线；节点用 _svg_node 生成的矢量图。
        交互（缩放/平移/WASD/还原/悬停）由内联 JS 实现。返回 HTML 路径。
        """
        import json

        depth = self._layers()
        real_pos, dummy_pos, edge_routes, level_sizes = self._compute_layout(depth)
        paths = self._build_edge_paths(real_pos, dummy_pos, edge_routes)

        node_w, node_h = 240.0, 86.0

        # 全图坐标范围（含节点半宽/半高留白）
        all_xs = [p[0] for p in real_pos.values()] + [p[0] for p in dummy_pos.values()]
        all_ys = [p[1] for p in real_pos.values()] + [p[1] for p in dummy_pos.values()]
        min_x, max_x = min(all_xs) - node_w / 2, max(all_xs) + node_w / 2
        min_y, max_y = min(all_ys) - node_h / 2, max(all_ys) + node_h / 2

        # 边（先画，被节点覆盖形成"从节点边缘出发"的观感）
        edges_svg = []
        for pts, color, _title in paths:
            d = self._netron_curve_path(pts)
            edges_svg.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
                f'marker-end="url(#arrow)"/>'
            )
        edges_svg = ''.join(edges_svg)

        # 节点 + tooltip 数据
        node_titles: list[str] = []
        nodes_svg = []
        for name in sorted(self.layer_row.keys(), key=lambda x: self.node_order[x]):
            row = self.layer_row[name]
            op = row.get('op_type', '?')
            image, title = self._svg_node(name, op, row)
            x, y = real_pos[name]
            idx = len(node_titles)
            node_titles.append(title)
            nodes_svg.append(
                f'<image href="{image}" x="{x - node_w / 2:.1f}" y="{y - node_h / 2:.1f}" '
                f'width="{node_w:.0f}" height="{node_h:.0f}" data-i="{idx}" class="node"/>'
            )
        nodes_svg = ''.join(nodes_svg)
        titles_json = json.dumps(node_titles)

        # 颜色条图例
        c_low, c_mid, c_high = '#ff4500', '#ffd700', '#3cb371'
        legend_html = (
            '<div id="legend">'
            '<div id="legendTitle"><span class="dot" style="background:#335588"></span>'
            '<b>entire_cos</b> 累积余弦精度</div>'
            '<div id="legendBar"></div>'
            '<div id="legendTicks">'
            '<span style="color:' + c_low + '">0.80</span>'
            '<span style="color:' + c_high + '">0.90</span>'
            '<span style="color:#2e8b57">1.00</span>'
            '</div>'
            '<div id="legendNote">悬停节点查看 <b>单层精度 / 欧氏距离</b></div>'
            '</div>'
        )

        reset_btn = (
            '<button id="btnReset" title="还原视图到初始布局" '
            'style="position:fixed;top:16px;left:16px;z-index:2000;'
            'padding:8px 14px;border:1px solid #bbb;border-radius:6px;'
            'background:#fff;color:#333;font-family:Arial;font-size:13px;'
            'cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.2);">'
            '&#x27f3; 还原布局</button>'
        )

        # 左下角缩放按钮（+ 放大 / - 缩小）
        zoom_btns = (
            '<div style="position:fixed;bottom:20px;left:16px;z-index:2000;'
            'display:flex;flex-direction:column;gap:6px;">'
            '<button id="btnZoomIn" title="放大" '
            'style="width:34px;height:34px;border:1px solid #bbb;border-radius:6px;'
            'background:#fff;color:#333;font-family:Arial;font-size:20px;font-weight:bold;'
            'line-height:1;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.2);">+</button>'
            '<button id="btnZoomOut" title="缩小" '
            'style="width:34px;height:34px;border:1px solid #bbb;border-radius:6px;'
            'background:#fff;color:#333;font-family:Arial;font-size:20px;font-weight:bold;'
            'line-height:1;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.2);">-</button>'
            '</div>'
        )

        # 全部用占位符，避免 f-string 与 JS 的 {} 冲突
        script = (
            '<script>\n'
            '(function() {\n'
            '  var svg = document.getElementById("svg");\n'
            '  var tip = document.getElementById("tip");\n'
            '  var B = {minX:__MINX__, maxX:__MAXX__, minY:__MINY__, maxY:__MAXY__};\n'
            '  var titles = __TITLES__;\n'
            '  var panSpeed = __PAN_SPEED__;\n'
            '  var initView = null;\n'
            '  function getView() {\n'
            '    var v = svg.getAttribute("viewBox").split(/[ ,]+/).map(Number);\n'
            '    return {x: v[0], y: v[1], w: v[2], h: v[3]};\n'
            '  }\n'
            '  function setView(x, y, w, h) { svg.setAttribute("viewBox", x + " " + y + " " + w + " " + h); renderMinimap(); }\n'
            '  // ---- 小地图（minimap）----\n'
            '  var mmSvg = document.getElementById("minimapSvg");\n'
            '  var mmW = 220, mmH = 400;\n'
            '  var mmData = null;  // {nodes:[{x,y}], scale, ox, oy}\n'
            '  function mmScale() {\n'
            '    // 等比缩放让全图 fit 进小地图（居中）\n'
            '    return Math.min(mmW / (B.maxX - B.minX), mmH / (B.maxY - B.minY));\n'
            '  }\n'
            '  function mmToWorld(px, py) {\n'
            '    // 小地图像素 -> 世界坐标\n'
            '    var s = mmScale();\n'
            '    var ox = (mmW - (B.maxX - B.minX) * s) / 2;\n'
            '    var oy = (mmH - (B.maxY - B.minY) * s) / 2;\n'
            '    return {x: B.minX + (px - ox) / s, y: B.minY + (py - oy) / s};\n'
            '  }\n'
            '  function renderMinimap() {\n'
            '    if (!mmData) { return; }\n'
            '    var vb = getView();\n'
            '    var s = mmScale();\n'
            '    var ox = (mmW - (B.maxX - B.minX) * s) / 2;\n'
            '    var oy = (mmH - (B.maxY - B.minY) * s) / 2;\n'
            '    var wx = function(x) { return ox + (x - B.minX) * s; };\n'
            '    var wy = function(y) { return oy + (y - B.minY) * s; };\n'
            '    mmSvg.innerHTML = "";\n'
            '    var ns = "http://www.w3.org/2000/svg";\n'
            '    // 所有节点小方块\n'
            '    mmData.nodes.forEach(function(n) {\n'
            '      var r = document.createElementNS(ns, "rect");\n'
            '      r.setAttribute("x", wx(n.x) - 2);\n'
            '      r.setAttribute("y", wy(n.y) - 2);\n'
            '      r.setAttribute("width", 4);\n'
            '      r.setAttribute("height", 4);\n'
            '      r.setAttribute("fill", "#6688aa");\n'
            '      r.setAttribute("rx", 1);\n'
            '      mmSvg.appendChild(r);\n'
            '    });\n'
            '    // 当前视窗框\n'
            '    var vr = document.createElementNS(ns, "rect");\n'
            '    vr.setAttribute("class", "mm-rect");\n'
            '    vr.setAttribute("x", wx(vb.x));\n'
            '    vr.setAttribute("y", wy(vb.y));\n'
            '    vr.setAttribute("width", Math.max(wx(vb.x + vb.w) - wx(vb.x), 2));\n'
            '    vr.setAttribute("height", Math.max(wy(vb.y + vb.h) - wy(vb.y), 2));\n'
            '    mmSvg.appendChild(vr);\n'
            '    mmData.rect = vr;\n'
            '  }\n'
            '  // 点击/拖拽小地图跳转\n'
            '  var mmDrag = false;\n'
            '  function mmJump(px, py) {\n'
            '    var pos = mmToWorld(px, py);\n'
            '    var vb = getView();\n'
            '    setView(pos.x - vb.w / 2, pos.y - vb.h / 2, vb.w, vb.h);\n'
            '  }\n'
            '  mmSvg.addEventListener("mousedown", function(e) {\n'
            '    mmDrag = true;\n'
            '    var r = mmSvg.getBoundingClientRect();\n'
            '    mmJump(e.clientX - r.left, e.clientY - r.top);\n'
            '    e.preventDefault(); e.stopPropagation();\n'
            '  });\n'
            '  window.addEventListener("mousemove", function(e) {\n'
            '    if (!mmDrag) { return; }\n'
            '    var r = mmSvg.getBoundingClientRect();\n'
            '    mmJump(e.clientX - r.left, e.clientY - r.top);\n'
            '  });\n'
            '  window.addEventListener("mouseup", function() { mmDrag = false; });\n'
            '  function initialView() {\n'
            '    var figH = (B.maxY - B.minY) || 1;\n'
            '    var targetH = figH / 10;\n'
            '    var vw = targetH * (window.innerWidth / window.innerHeight);\n'
            '    var vh = targetH;\n'
            '    var cx = (B.minX + B.maxX) / 2;\n'
            '    var cy = B.minY + targetH * 0.4;\n'
            '    return {x: cx - vw / 2, y: cy - vh / 2, w: vw, h: vh};\n'
            '  }\n'
            '  function zoomAt(px, py, factor) {\n'
            '    var vb = getView();\n'
            '    var cx = vb.x + (px / window.innerWidth) * vb.w;\n'
            '    var cy = vb.y + (py / window.innerHeight) * vb.h;\n'
            '    var nw = vb.w * factor, nh = vb.h * factor;\n'
            '    setView(cx - (px / window.innerWidth) * nw, cy - (py / window.innerHeight) * nh, nw, nh);\n'
            '  }\n'
            '  svg.addEventListener("wheel", function(e) {\n'
            '    e.preventDefault();\n'
            '    zoomAt(e.clientX, e.clientY, e.deltaY > 0 ? 1.15 : 1 / 1.15);\n'
            '  }, {passive: false});\n'
            '  var dragging = false, sx = 0, sy = 0, svx = 0, svy = 0;\n'
            '  svg.addEventListener("mousedown", function(e) {\n'
            '    dragging = true; sx = e.clientX; sy = e.clientY;\n'
            '    var vb = getView(); svx = vb.x; svy = vb.y;\n'
            '    svg.style.cursor = "grabbing";\n'
            '  });\n'
            '  window.addEventListener("mousemove", function(e) {\n'
            '    if (!dragging) { return; }\n'
            '    var vb = getView();\n'
            '    var dx = (e.clientX - sx) / window.innerWidth * vb.w;\n'
            '    var dy = (e.clientY - sy) / window.innerHeight * vb.h;\n'
            '    setView(svx - dx, svy - dy, vb.w, vb.h);\n'
            '  });\n'
            '  window.addEventListener("mouseup", function() { dragging = false; svg.style.cursor = "grab"; });\n'
            '  var pressed = {};\n'
            '  window.addEventListener("keydown", function(e) { pressed[e.key.toLowerCase()] = true; });\n'
            '  window.addEventListener("keyup", function(e) { pressed[e.key.toLowerCase()] = false; });\n'
            '  function pan(dx, dy) {\n'
            '    var vb = getView();\n'
            '    setView(vb.x + dx, vb.y + dy, vb.w, vb.h);\n'
            '  }\n'
            '  (function loop() {\n'
            '    if (pressed["w"]) { pan(0, -panSpeed); }\n'
            '    if (pressed["s"]) { pan(0, panSpeed); }\n'
            '    if (pressed["a"]) { pan(-panSpeed, 0); }\n'
            '    if (pressed["d"]) { pan(panSpeed, 0); }\n'
            '    requestAnimationFrame(loop);\n'
            '  })();\n'
            '  svg.addEventListener("mousemove", function(e) {\n'
            '    if (pinnedIdx >= 0) { return; }\n'
            '    var t = e.target; if (!t || !t.classList || !t.classList.contains("node")) { tip.style.display = "none"; return; }\n'
            '    var i = Number(t.getAttribute("data-i"));\n'
            '    if (titles[i]) { tip.textContent = titles[i]; tip.style.display = "block"; }\n'
            '    tip.style.left = (e.clientX + 14) + "px";\n'
            '    tip.style.top = (e.clientY + 14) + "px";\n'
            '  });\n'
            '  svg.addEventListener("mouseleave", function() { if (pinnedIdx < 0) { tip.style.display = "none"; } });\n'
            '  var pinnedIdx = -1;\n'
            '  // 点击节点固定信息卡片；点击别处关闭。卡片内点击（选择/复制）不关闭。\n'
            '  document.addEventListener("click", function(e) {\n'
            '    var t = e.target;\n'
            '    if (t && t.id === "tip") { return; }\n'
            '    if (t && t.classList && t.classList.contains("node")) {\n'
            '      pinnedIdx = Number(t.getAttribute("data-i"));\n'
            '      tip.textContent = titles[pinnedIdx];\n'
            '      tip.classList.add("pinned");\n'
            '      tip.style.display = "block";\n'
            '      tip.style.left = (e.clientX + 14) + "px";\n'
            '      tip.style.top = (e.clientY + 14) + "px";\n'
            '      return;\n'
            '    }\n'
            '    pinnedIdx = -1;\n'
            '    tip.classList.remove("pinned");\n'
            '    tip.style.display = "none";\n'
            '  });\n'
            '  window.__resetLayout = function() { if (initView) setView(initView.x, initView.y, initView.w, initView.h); };\n'
            '  var btn = document.getElementById("btnReset");\n'
            '  if (btn) { btn.addEventListener("click", function() { window.__resetLayout(); }); }\n'
            '  var zi = document.getElementById("btnZoomIn");\n'
            '  if (zi) { zi.addEventListener("click", function() { zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.15); }); }\n'
            '  var zo = document.getElementById("btnZoomOut");\n'
            '  if (zo) { zo.addEventListener("click", function() { zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.15); }); }\n'
            '  // 收集节点真实坐标，供小地图使用\n'
            '  mmData = {nodes: []};\n'
            '  qs = svg.querySelectorAll("image.node");\n'
            '  for (var qi = 0; qi < qs.length; qi++) {\n'
            '    var q = qs[qi];\n'
            '    mmData.nodes.push({x: parseFloat(q.getAttribute("x")) + parseFloat(q.getAttribute("width")) / 2,\n'
            '                        y: parseFloat(q.getAttribute("y")) + parseFloat(q.getAttribute("height")) / 2});\n'
            '  }\n'
            '  initView = initialView();\n'
            '  setView(initView.x, initView.y, initView.w, initView.h);\n'
            '})();\n'
            '</script>'
        )
        script = (script.replace('__MINX__', f'{min_x:.1f}')
                       .replace('__MAXX__', f'{max_x:.1f}')
                       .replace('__MINY__', f'{min_y:.1f}')
                       .replace('__MAXY__', f'{max_y:.1f}')
                       .replace('__TITLES__', titles_json)
                       .replace('__PAN_SPEED__', str(self.pan_speed)))

        html = (
            '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
            f'<title>{self.title}</title>\n'
            '<style>\n'
            'html,body{margin:0;height:100%;overflow:hidden;font-family:Arial;}\n'
            'svg#svg{display:block;width:100vw;height:100vh;background:#fff;cursor:grab;}\n'
            'image.node{cursor:pointer;}\n'
            '#tip{position:fixed;display:none;z-index:3000;background:rgba(255,255,255,.97);'
            'border:1px solid #aaa;border-radius:4px;padding:6px 10px;font-size:12px;'
            'color:#222;white-space:pre;box-shadow:0 1px 4px rgba(0,0,0,.2);pointer-events:none;max-width:520px;}\n'
            '#tip.pinned{pointer-events:auto;cursor:text;user-select:text;}\n'
            '#legend{position:fixed;bottom:20px;right:20px;z-index:2000;background:#fff;'
            'border:1px solid #d8dde4;border-radius:8px;padding:12px 14px;font-family:Arial;'
            'font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);'
            'box-sizing:border-box;width:234px;}\n'
            '#legendTitle{display:flex;align-items:center;gap:6px;font-size:13px;'
            'color:#333;margin-bottom:8px;}\n'
            '#legendTitle .dot{width:12px;height:12px;border-radius:3px;display:inline-block;}\n'
            '#legendBar{width:100%;height:14px;border-radius:7px;'
            'background:linear-gradient(to right, ' + c_low + ', ' + c_mid + ', ' + c_high + ');}\n'
            '#legendTicks{display:flex;justify-content:space-between;margin-top:3px;'
            'font-size:11px;font-weight:bold;}\n'
            '#legendNote{margin-top:9px;padding-top:8px;border-top:1px solid #eee;'
            'color:#888;font-size:11px;line-height:1.5;}\n'
            '#minimap{position:fixed;top:16px;right:20px;z-index:2000;background:#fff;'
            'border:1px solid #bbb;border-radius:6px;padding:6px;box-shadow:0 1px 4px rgba(0,0,0,.2);'
            'user-select:none;cursor:crosshair;}\n'
            '#minimap svg{display:block;background:#f8f8f8;border-radius:4px;width:220px;height:400px;}\n'
            '#minimap .mm-rect{fill:rgba(60,179,113,.12);stroke:#3cb371;stroke-width:1.5;}\n'
            '</style>\n</head>\n<body>\n'
            '<svg id="svg" preserveAspectRatio="xMidYMid meet">\n'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#666"/></marker></defs>\n'
            + edges_svg + nodes_svg +
            '</svg>\n'
            + reset_btn + zoom_btns + legend_html +
            '<div id="minimap"><svg id="minimapSvg" width="220" height="400"></svg></div>\n'
            '<div id="tip"></div>\n'
            + script +
            '\n</body>\n</html>\n'
        )

        self.output_path.write_text(html, encoding='utf-8')

        print(f"Network analysis HTML saved to: {self.output_path}")
        if show:
            import webbrowser
            webbrowser.open(self.output_path.as_uri())
        return str(self.output_path)

# ----------------------------------------------------------------------
# CLI：需 SDK 的纯函数子命令，供 QnnAccuracyDebugger 以子进程调用
#   python accuracy_debugger.py encodings <dlc> <out_json>
#   python accuracy_debugger.py layers    <dlc> <out_json>
# ----------------------------------------------------------------------

def _main(argv=None) -> int:
    import sys as _s
    argv = list(_s.argv[1:] if argv is None else argv)
    if not argv:
        print('usage: accuracy_debugger.py encodings|layers <dlc> <out_json>'
              '| tinydlc <dlc> <out_json> | tinydlc-batch <dlcs_list_json>',
              file=_s.stderr)
        return 2
    subcmd = argv[0]
    if subcmd == 'tinydlc-batch':
        try:
            QnnTruncatedAccuracyAnalysis.dump_tinydlc_meta_batch(argv[1])
            print('tinydlc-batch done')
            return 0
        except Exception as exc:
            print(f'[accuracy_debugger CLI] tinydlc-batch failed: {type(exc).__name__}: {exc}',
                  file=_s.stderr)
            return 1
    if len(argv) < 3:
        print('usage: accuracy_debugger.py encodings|layers <dlc> <out_json>'
              '| tinydlc <dlc> <out_json>', file=_s.stderr)
        return 2
    dlc, out_json = argv[1], argv[2]
    try:
        if subcmd == 'encodings':
            DlcV2EncodingExporter(dlc).dump(out_json)
        elif subcmd == 'layers':
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(DlcV2EncodingExporter(dlc).layer_candidates(),
                          f, ensure_ascii=False, indent=1)
            print(f"layers -> {out_json}")
        elif subcmd == 'tinydlc':
            QnnTruncatedAccuracyAnalysis.dump_tinydlc_meta(dlc, out_json)
            print(f"tinydlc -> {out_json}")
        else:
            print(f'unknown subcommand: {subcmd}', file=_s.stderr)
            return 2
    except Exception as exc:  # 子进程内失败：非零退出即可（父进程看返回码）
        print(f'[accuracy_debugger CLI] {subcmd} failed: {type(exc).__name__}: {exc}',
              file=_s.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
