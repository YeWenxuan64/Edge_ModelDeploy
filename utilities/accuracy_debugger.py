import re
import os
import sys
import csv
import json
import html
import runpy
import shlex
import shutil
import subprocess
import traceback

from pathlib import Path
from datetime import datetime
from typing import Callable
from collections import defaultdict, deque

import numpy as np
import cv2
import onnx
from onnx import helper, TensorProto


current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
from utils import clean_files_or_dirs, run_command, MultThreadExetutor




# 层精度 schema（内存 dict 与 CSV 共用同一套键；历史名 LAYER_ACC_KEYS）
#   键序 = CSV 列序 = 渲染 x 轴序：
#     entire_cos, entire_euc, entire_mse, single_cos, single_euc
#   dict 形态：{输出张量名(sanitized): {上述键: float|None}}；界面层名另取生产算子名。
#     - entire 侧有值 = 该层在累积(entire)分析中有结果；
#     - single 侧有值 = 该层在单层(single/truncated)分析中有结果；
#     - 无结果侧一律 None（CSV 空列读回亦为 None），由下游自行解析；
#     - 单层分析不产出 single_mse。
#   内存可选附加键（非 CSV 列）：op_type 算子类型 —— Rknn 侧由 error_analysis
#   自带类型带入；Qnn 侧缺失时 AccuracyGraph 回退 onnx node_info 推断。


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

    def _acc_col(accuracy: dict, key: str) -> list:
        """从统一 dict 按层序取某指标列（缺失 None -> 原样保留，绘图处再转 nan）。"""
        return [row.get(key) if isinstance(row, dict) else None for row in accuracy.values()]

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
            - node_labels (dict[str, str]): 张量 -> 生产算子名，多输出附输出序号
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
    node_labels: dict[str, str] = {}
    pred: dict[str, list[str]] = {}
    succ: dict[str, list[str]] = {}
    for n in g.node:
        outs = [_norm(o) for o in n.output if o]
        ins = [_norm(i) for i in n.input if i]
        for index, original_out in enumerate(n.output):
            if not original_out:
                continue
            out = _norm(original_out)
            node_info.setdefault(out, (n.op_type, ins))
            node_labels.setdefault(out, (n.name + (f' [输出 {index}]' if len(outs) > 1 else '')) if n.name else '')
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
        'node_labels': node_labels,
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
        tmp_model_path (Path): convert() 复制到 tmp 目录的模型副本，图结构分析的回退参考图。
        snapshot_dir (Path): RKNN 精度分析快照目录（snapshot/）。
        fused_model_path (Path): RKNN fuse 阶段 dump 到本目录的内部图，图结构分析的首选参考图。
    """

    def __init__(self, tmp_work_dir:str, tmp_model_path:str):
        self.tmp_work_dir = Path(tmp_work_dir).resolve()
        self.tmp_model_path = Path(tmp_model_path).resolve()

        self.snapshot_dir = self.tmp_work_dir / 'snapshot'
        # error_analysis.txt 的层名来自这张图而不是入参 ONNX 副本：fuse 阶段会把
        # GlobalAveragePool 之类算子改写成 <节点名>_2convN 等合成层，只有末级沿用
        # 原来的 <...>_output_0 张量名。是否可用要到 convert() 后才能判定（本类在
        # set_do_accuracy_analysis() 里就构造好了），所以这里只记路径。
        self.fused_model_path = self.tmp_work_dir / 'check3_fuse_ops.onnx'

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
        完成一次）。RKNN 快照已含输入层，include_input=False；层名为 RKNN 内部图
        的张量名（未清洗），sanitize=False、layer_display 恒等，故结构参考图优先用
        fused_model_path，缺失时回退 tmp_model_path。
        """

        accuracy = self.read_error_analysis()
        if not accuracy:
            print("No data to plot.")
            return 

        save_path = self.tmp_work_dir / 'rknn_accuracy_analysis_summary.png'
        plot_accuracy_summary(accuracy, entire_val_color="orange", save_path=save_path)
        self.file_or_dir_to_clean.append(save_path)


        output_path = self.tmp_work_dir / 'rknn_graph_accuracy_analysis.html'
        graph_model_path = self.fused_model_path if self.fused_model_path.exists() else self.tmp_model_path

        viz = AccuracyGraph(
            accuracy,
            graph_model_path,
            output_path,
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

    backend 选择（实测）：
      - Linux：FP32 -> CPU 后端，量化 -> HTP 后端（CPU 拒绝量化/16bit 图）；
      - Windows x86：两侧都 QnnCpu.dll（无 HTP 执行环境），故 single 不可用；
      - Windows ARM64：推理侧切原生 ARM64（见 _prepare_native_net_run）。
    """

    # WoA 上推理侧（net-run / ctx-gen）用的 SDK 子目录；转换侧仍是 x86 工具
    WIN_ARM64_ARCH = 'aarch64-windows-msvc'

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

        self.native_net_run: bool = False   # 推理侧是否已切原生 ARM64
        self.native_htp_config: Path | None = None   # 实测宿主 DSP 架构，不沿用部署目标 SoC

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


    def _prepare_native_net_run(self):
        """WoA 上把推理侧切到原生 ARM64，其它平台原样返回。

        只切推理侧：converter/quantizer 只有 x86 版；而 16bit/fp16 的 DLC 在 x86 的
        qnn-net-run 上会卡死在 "Executing Graphs"，原生 ARM64 才能驱动 HTP。
        裸名工具按 PATH 决定架构，故只需前置 ARM64 的 bin/lib，另加 skel 搜索路径
        （ADSP_LIBRARY_PATH，缺了 DSP 打不开 session）。skel 按全部版本拼接：DLC 的
        编译目标与真机架构可以不一致，而它是搜索路径列表、名字自带版本号不冲突。
        """
        from onnx_to_qnn import QAIRTScript     # 惰性导入：见 QnnSdkTools.converter_script
        if not QAIRTScript.is_windows_arm64():
            return

        sdk_root = os.environ.get('QAIRT_SDK_ROOT')
        sdk = Path(sdk_root) if sdk_root else None
        bin_dir = sdk / 'bin' / self.WIN_ARM64_ARCH if sdk else None
        lib_dir = sdk / 'lib' / self.WIN_ARM64_ARCH if sdk else None
        if not (bin_dir and bin_dir.is_dir() and lib_dir.is_dir()):
            print(f"[QnnAccuracyDebugger] Windows ARM64 host but the SDK has no "
                  f"{self.WIN_ARM64_ARCH}; keeping the x86 net-run environment")
            return

        os.environ['Path'] = os.pathsep.join(
            [str(bin_dir), str(lib_dir), os.environ.get('Path', '')])

        skel_dirs = sorted((sdk / 'lib').glob('hexagon-v*/unsigned'))
        if skel_dirs:
            os.environ['ADSP_LIBRARY_PATH'] = os.pathsep.join(str(d) for d in skel_dirs)

        # 量化侧走 HTP（与 Linux 同策略）；golden 仍留在 CPU
        self.backend_lib_target = 'QnnHtp.dll'
        self.native_net_run = True
        print(f"[QnnAccuracyDebugger] native ARM64 net-run: {len(skel_dirs)} skel dirs, "
              f"target backend -> {self.backend_lib_target}")
        self._prepare_native_htp_config(bin_dir, lib_dir)

    def _prepare_native_htp_config(self, bin_dir: Path, lib_dir: Path) -> None:
        """检测实际 DSP 架构，避免 context 编译默认按 v68 拒绝 FP16 子图。"""
        self.native_htp_config = None
        validator = bin_dir / 'qnn-platform-validator.exe'
        if not validator.is_file():
            return
        config_dir = self.working_dir / 'native_htp'
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / 'validator').mkdir(exist_ok=True)
        try:
            probe = subprocess.run(
                [str(validator), '--backend', 'dsp', '--coreVersion',
                 '--targetPath', str(config_dir / 'validator')],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=15, env=os.environ)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (config_dir / 'validator.log').write_text(str(exc), encoding='utf-8')
            return
        output = (probe.stdout or '') + (probe.stderr or '')
        (config_dir / 'validator.log').write_text(output, encoding='utf-8')
        arch = re.search(r'Core Version of the backend DSP:\s*Hexagon Architecture V(\d+)',
                         output, re.IGNORECASE)
        if probe.returncode != 0 or arch is None:
            return
        dsp_arch = f'v{arch.group(1)}'
        backend_path = config_dir / 'backend.json'
        backend_path.write_text(json.dumps({'devices': [{'dsp_arch': dsp_arch}]}, indent=2),
                                encoding='utf-8')
        config_path = config_dir / 'config.json'
        config_path.write_text(json.dumps({'backend_extensions': {
            'shared_library_path': str(lib_dir / 'QnnHtpNetRunExtensions.dll'),
            'config_file_path': str(backend_path),
        }}, indent=2), encoding='utf-8')
        self.native_htp_config = config_path
        print(f"[QnnAccuracyDebugger] native HTP architecture: {dsp_arch}; "
              f"config: {config_path}")

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
        if self.native_htp_config and backend_lib == self.backend_lib_target:
            cmd += f' --config_file "{self.native_htp_config}"'

        log_path = Path(output_dir) / 'net-run.log'
        print(f"[QAIRTAccuracyDebugger] net-run {backend_lib}; log: {log_path}")
        return run_command(cmd, signature="[QAIRTAccuracyDebugger]", print_output=False,
                           log_path=log_path)

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
        self._prepare_native_net_run()          # WoA 上切原生 ARM64，须早于任何 net-run
        self.target_dlc_path = str(target_dlc_path)
        input_list = self.prepare_input_data(mean_rgb, std_rgb)

        # 1) entire（累积）——末尾 save_accuracy_analysis_csv 落盘
        self.entire_accuracy_analysis(golden_dlc_path, target_dlc_path, input_list)

        # 2) single（单层）——失败时退回 entire-only 渲染（不阻断既有流程）。
        #    成功则 save_accuracy_analysis_csv(kind='single') 追加到同一 CSV。
        #    Windows x86 无 HTP、single 必失败故跳过；WoA 已切原生 ARM64 则放行。
        fallback_entire_only = sys.platform.startswith('win') and not self.native_net_run

        if not fallback_entire_only:
            try:
                self.single_accuracy_analysis(golden_dlc_path, target_dlc_path, input_list)
            except Exception as exc:
                fallback_entire_only = True
                failure_log = self.working_dir / 'single_failures.log'
                try:
                    with failure_log.open('a', encoding='utf-8') as log:
                        log.write(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                                  f"SDK={os.environ.get('QAIRT_SDK_ROOT', '')}\n"
                                  f"golden={golden_dlc_path}\ntarget={target_dlc_path}\n"
                                  f"{traceback.format_exc()}")
                except OSError as log_exc:
                    print(f"[QAIRTAccuracyDebugger] cannot save single failure: {log_exc}")
                print(f"[QAIRTAccuracyDebugger] single-layer analysis skipped ({type(exc).__name__}: {exc}); "
                      f"fallback to entire-only report; log: {failure_log}")


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
        try:
            accuracy = analyzer.run(golden_dlc_path, target_dlc_path, input_list, only_tensor_prefix)
        finally:
            analyzer.save_stage_results()
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
            sanitize=True,
            include_input=True,
            layer_display=lambda g, k: g['tensors'].get(k, k),
            dlc_candidates=dlc_candidates,
            dlc_io=dlc_io,
        )
        html_path = viz.render(show=show)
        self.file_or_dir_to_clean.append(html_path)
        return viz.data


# ============ QNN SDK 工具 / DLC 反射 / 逐层分析（与 RKNN 侧无关的 QNN 分区） ============

class QnnSdkTools:
    """QAIRT SDK 侧的调用与读取工具（无状态，全部 classmethod / staticmethod）。

    与"精度分析算法"无关、纯粹对接 SDK 的部分集中在这里：SDK pythonpath 注入、
    QNN_DATATYPE_* -> dtype 名、qairt-converter 调用（含进程池 worker）、tiny DLC
    meta 直读与落盘。消费方：QnnTruncatedAccuracyAnalysis、DlcV2EncodingExporter、
    本文件的 CLI 子命令。
    """

    # QNN_DATATYPE_UFIXED_POINT_8 / SFIXED_POINT_16 / FLOAT_16 ... -> 'uint8' / 'int16' / 'float16'
    dtype_name_patterns = (
        (re.compile(r'UFIXED_POINT_(\d+)'), 'uint'),
        (re.compile(r'SFIXED_POINT_(\d+)'), 'int'),
        (re.compile(r'FLOAT_(\d+)'), 'float'),
    )
    # dtype 名 -> numpy 类型：读 qnn-net-run dump 时用它的位宽判断文件字节数，
    # 8/16bit 整型与 fp16 激活都要覆盖
    numpy_dtype_of_qnn_dtype = {
        'float16': np.float16, 'float32': np.float32,
        'int8': np.int8, 'uint8': np.uint8,
        'int16': np.int16, 'uint16': np.uint16,
        'int32': np.int32, 'uint32': np.uint32,
    }

    converter_script_path = None        # qairt-converter 脚本绝对路径（进程内缓存一次）

    # ------------------------------------------------------------------
    # SDK 环境 / dtype 名称
    # ------------------------------------------------------------------

    @staticmethod
    def inject_pythonpath() -> None:
        """主进程内使用 qti 前，把 PYTHONPATH 里的 SDK python 目录并入 sys.path。

        os.environ 的 PYTHONPATH 只在解释器启动时读取，run_env_script 运行时
        注入的环境不会刷新已运行解释器的 sys.path。
        """
        for path_entry in os.environ.get('PYTHONPATH', '').split(os.pathsep):
            if path_entry and path_entry not in sys.path:
                sys.path.insert(0, path_entry)

    @classmethod
    def dtype_name(cls, qnn_data_type) -> str:
        """QNN_DATATYPE_* 枚举 -> 'uint8'/'int16'/'float32'；不匹配任何模式时原样返回枚举名。"""
        dtype_text = getattr(qnn_data_type, 'name', None) or str(qnn_data_type)
        for pattern, bit_prefix in cls.dtype_name_patterns:
            matched = pattern.search(dtype_text)
            if matched:
                return bit_prefix + matched.group(1)
        return dtype_text

    # ------------------------------------------------------------------
    # converter：脚本定位 / 命令行 / 进程池 worker
    # ------------------------------------------------------------------

    @classmethod
    def converter_script(cls) -> str:
        """qairt-converter 脚本绝对路径。

        get_tool 在 Windows 返回「解释器 + 脚本」两个带引号 token（Linux 只有一个），
        故按引号切分取最后一个；早先的 strip('"') 会把两者粘成非法路径。
        """
        if cls.converter_script_path is None:
            from onnx_to_qnn import QAIRTScript
            cls.converter_script_path = shlex.split(
                QAIRTScript.get_tool('qairt-converter'))[-1]
        return cls.converter_script_path

    @staticmethod
    def converter_argv(cut_onnx: str, out_dlc: str, ov_json: str, layout: str) -> list:
        """cut 图 + 单层 overrides -> qairt-converter 参数；layout 串用 shlex.split 去引号。"""
        args = ['--input_network', cut_onnx, '--output_path', out_dlc,
                '--quantization_overrides', ov_json]
        args += shlex.split(layout)
        args += ['--onnx_skip_simplification']
        return args

    @staticmethod
    def convert_one(job: tuple) -> tuple:
        """进程池 worker：转换一个 cut 图 -> 量化 DLC，返回 (SDK rc, 失败原因)。

        job = (cut_onnx, ov_json, layout, out_dlc)（均为 str 路径）。
        runpy 让 worker 复用重模块；C 层直写 fd1/fd2，完整输出按层写入 converter.log。
        保持 staticmethod + 类名显式取工具：进程池按限定名 pickle 本函数。
        """
        cut, ov, layout, out_dlc = job
        QnnSdkTools.inject_pythonpath()
        script = QnnSdkTools.converter_script()
        argv = [script] + QnnSdkTools.converter_argv(cut, out_dlc, ov, layout)
        old_argv = sys.argv
        sys.argv = argv
        diagnostic, exception_trace = '', ''
        log_path = Path(out_dlc).parent / 'converter.log'
        try:
            with log_path.open('w+', encoding='utf-8', errors='replace') as tf:
                try:
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
                finally:
                    tf.flush()
            rc = 0
        except SystemExit as e:
            rc = int(e.code or 0)
        except Exception as exc:
            diagnostic = f'{type(exc).__name__}: {exc}'
            exception_trace = traceback.format_exc()
            rc = None
        finally:
            sys.argv = old_argv
        if rc == 0:
            # 池崩时 Future 返回值可能丢失；只有 converter 正常结束才留下完成标记。
            try:
                out = Path(out_dlc)
                if not out.is_file() or out.stat().st_size == 0:
                    diagnostic = 'converter returned success without a non-empty DLC'
                else:
                    out.with_suffix('.converted').touch()
            except OSError as exc:
                diagnostic = f'cannot mark converted DLC: {exc}'
        try:
            with log_path.open('a', encoding='utf-8') as log:
                log.write(f'\nconverter rc={rc}; status={"failed" if diagnostic or rc else "ok"}\n')
                if diagnostic:
                    log.write(diagnostic + '\n')
                if exception_trace:
                    log.write(exception_trace)
        except OSError:
            pass  # 转换和完成标记已确认，附属日志失败不撤销有效产物。
        return rc, diagnostic

    @staticmethod
    def convert_one_isolated(job: tuple) -> tuple:
        """隔离子进程转换一个 cut 图 -> (SDK rc, 失败原因)；仅在池崩后补跑少量层时用。

        与 convert_one 的区别：每层开全新进程（放弃引擎复用），因为 worker 的
        0xC0000005 会连带杀掉同进程里后续的层，独立进程只损失该层、可重试。
        job 含义同 convert_one；环境用 os.environ（x86 转换侧）。
        """
        cut, ov, layout, out_dlc = job
        argv = ([sys.executable, QnnSdkTools.converter_script()]
                + QnnSdkTools.converter_argv(cut, out_dlc, ov, layout))
        proc = subprocess.run(argv, env=os.environ, capture_output=True, text=True,
                              errors='replace')
        tail = ((proc.stderr or '') + (proc.stdout or '')).strip()
        rc, reason = proc.returncode, ''
        out = Path(out_dlc)
        if rc == 0 and (not out.is_file() or out.stat().st_size == 0):
            reason = 'converter returned success without a non-empty DLC'
        try:
            (Path(out_dlc).parent / 'converter_retry.log').write_text(
                f'{subprocess.list2cmdline(argv)}\n{tail}\nconverter rc={rc}\n{reason}\n',
                encoding='utf-8')
        except OSError:
            pass  # 日志写入失败不撤销 SDK 已确认的有效 DLC。
        return rc, reason

    # ------------------------------------------------------------------
    # tiny DLC meta 直读（qti IR 图）
    # ------------------------------------------------------------------
    # 每个文件新建 IrDlcReader：复用同一 reader 重复 open 会返回旧图导致串数据。

    @classmethod
    def read_tiny_meta(cls, dlc_path: str) -> dict:
        """tiny DLC -> {'inputs', 'outputs', 'tensors'}；tensors 项含 dtype/dims，
        量化张量附 scale（axis 量化再带 axis）与 zp（QNN 约定为负，offset 0 时省略）。
        """
        cls.inject_pythonpath()
        from qti.aisw.dlc_utils import modeltools
        reader = modeltools.IrDlcReader()
        reader.open(str(dlc_path))
        ir_graph = reader.get_ir_graph()

        def collect_boundary_tensors(boundary_getter):
            """图边界张量列表；老/新 SDK 的 getter 名字不同，取不到就当空。"""
            try:
                return [{'name': ir_tensor.name(), 'dims': list(ir_tensor.dims()),
                         'dtype': cls.dtype_name(ir_tensor.data_type())}
                        for ir_tensor in getattr(ir_graph, boundary_getter)()]
            except Exception:
                return []

        graph_inputs = (collect_boundary_tensors('get_input_tensors_to_graph')
                        or collect_boundary_tensors('get_input_tensors'))
        graph_outputs = (collect_boundary_tensors('get_output_tensors_of_graph')
                         or collect_boundary_tensors('get_output_tensors'))
        tensors = {}
        for tensor_name, ir_tensor in ir_graph.get_tensor_map().items():
            tensor_entry = {'dtype': cls.dtype_name(ir_tensor.data_type()),
                            'dims': list(ir_tensor.dims())}
            try:
                encoding = ir_tensor.get_encoding()
                if encoding is not None:
                    encoding_type_name = (getattr(getattr(encoding, 'type', None), 'name', None)
                                          or str(getattr(encoding, 'type', '')))
                    if 'AXIS_SCALE_OFFSET' in encoding_type_name:   # per-channel 量化
                        axis_enc_infos = encoding.axisEncInfo.encInfos
                        tensor_entry['scale'] = [enc_info.scale for enc_info in axis_enc_infos]
                        tensor_entry['axis'] = encoding.axisEncInfo.axis
                        if not all(enc_info.offset == 0 for enc_info in axis_enc_infos):
                            tensor_entry['zp'] = [enc_info.offset for enc_info in axis_enc_infos]
                    elif 'SCALE_OFFSET' in encoding_type_name:      # per-tensor 量化
                        tensor_entry['scale'] = encoding.encInfo.scale
                        if encoding.encInfo.offset != 0:
                            tensor_entry['zp'] = encoding.encInfo.offset   # QNN 约定为负
            except Exception:
                pass
            tensors[tensor_name] = tensor_entry
        return {'inputs': graph_inputs, 'outputs': graph_outputs, 'tensors': tensors}

    @classmethod
    def write_tiny_meta(cls, dlc_path: str, meta: dict | None = None,
                        out_json: str | None = None) -> str:
        """tiny DLC meta 落盘（默认 <stem>.tiny_meta.json，out_json 可另指），返回输出路径。

        meta 已读到手上就传入，避免同一个 DLC 解析两遍。
        """
        out = (Path(out_json) if out_json
               else Path(dlc_path).with_suffix('.tiny_meta.json'))
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(meta if meta is not None else cls.read_tiny_meta(dlc_path),
                      f, indent=1)
        return str(out)

    @classmethod
    def write_tiny_meta_batch(cls, dlcs_json: str) -> int:
        """批量落盘多个 tiny DLC 的 meta（CLI: tinydlc-batch），dlcs_json = {"dlcs": [绝对路径...]}。"""
        with open(dlcs_json, 'r', encoding='utf-8') as f:
            dlc_list = json.load(f)['dlcs']
        for dlc in dlc_list:
            try:
                cls.write_tiny_meta(dlc)
            except Exception as exc:
                print(f'[tinydlc-batch] failed {dlc}: {type(exc).__name__}: {exc}')
        return 0



# ============ 量化 DLC 反射器：encodings + 层候选 一站式读取 ============

class DlcV2EncodingExporter:
    """从量化 DLC 提取张量量化编码（AIMET 2.0.0）并反射层候选，一次开文件。

    qti 惰性导入：只有实例化时才要求 SDK python 路径可用（onnx_to_qnn.run_env_script
    注入环境，QnnSdkTools.inject_pythonpath 把 SDK python 目录并入 sys.path），
    因此本类可以在已具备 SDK sys.path / LD_LIBRARY_PATH 的解释器里直接实例化。
    单层分析（QnnTruncatedAccuracyAnalysis）主进程直接实例化本类读 encodings /
    layer-candidates，不再递归子进程。

    提供两类数据（同一 reader / 同一 IR 图）：
      - encodings：可量化张量的量化编码，导出为 AIMET 2.0.0 json
        （extract / to_dict / dump）；
      - layer candidates：每个 op 的每个输出 -> {tensor, op_type, inputs, outputs, io_tensors}
        （layer_candidates，另保留 op_name，供显示层名及单层 overrides 筛选）。

    qti 模块在首次实例化时惰性导入并缓存在类属性（进程内一次）；编码类型常量
    亦缓存到类上，避免逐张量重复属性查找。dtype 名称转换共用 QnnSdkTools。
    """

    # qti 惰性导入缓存（类级，首次 _ensure_sdk 时填充）。
    # 注意与实例属性 self._ir_graph（打开后的图对象）区分：下面两个是 qti 模块本身
    modeltools_module = None      # qti.aisw.dlc_utils.modeltools
    ir_graph_module = None        # qti.aisw.converters.common.ir_graph

    def __init__(self, dlc_path, reader=None):
        """打开量化 DLC 并读取 IR 图；reader 可注入（测试用），默认建 IrDlcReader。"""
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
        if cls.ir_graph_module is None:
            QnnSdkTools.inject_pythonpath()
            from qti.aisw.dlc_utils import modeltools
            from qti.aisw.converters.common import ir_graph
            cls.modeltools_module, cls.ir_graph_module = modeltools, ir_graph
            # encoding.type 判定常量（QNN_QUANTIZATION_ENCODING_*）缓存到类上，
            # 避免逐张量经模块属性查找
            cls._ENC_SCALE_OFFSET = ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            cls._ENC_BW_SCALE_OFFSET = ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET
            cls._ENC_AXIS_SCALE_OFFSET = ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
            cls._ENC_BW_AXIS_SCALE_OFFSET = ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
        return cls.modeltools_module, cls.ir_graph_module

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
        out = {'output_dtype': QnnSdkTools.dtype_name(ir_tensor.data_type()),
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
        """写出 encodings json（同官方 ModelEncoding.dump：indent=4, ensure_ascii=False），返回 out_json。"""
        data = self.to_dict()
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"exported {len(data['encodings'])} encodings -> {out_json}")
        return out_json

    # ------------------------------------------------------------------
    # 层候选反射（每算子 io 张量；供单层 overrides）
    # ------------------------------------------------------------------

    def layer_candidates(self) -> list[dict]:
        """每个 op 的每个输出各有一个候选；保留 op_name 与显式 inputs/outputs。

        io_tensors 范围与官方 subgraph_snooper 的 overrides 一致：Conv2d 取
        input+weight+bias+output，Prelu 取 input+coeff+output，Concat 取各输入激活+output。
        """
        candidates = []
        for op in self._ir_graph.get_ops():
            outs = [t.name() for t in op.outputs()]
            if not outs:
                continue
            ins = [t.name() for t in op.inputs()]
            op_type = str(op.type.name if hasattr(op.type, 'name') else op.type)
            for out in outs:
                candidates.append({
                    'tensor': out, 'op_type': op_type, 'op_name': getattr(op, 'name', '') or '',
                    'inputs': ins, 'outputs': outs,
                    'io_tensors': ins + outs,
                })
        return candidates

    def graph_io_tensors(self) -> dict:
        """IR 图声明的 {'inputs','outputs'} 张量名（权重/偏置等静态张量不在其中）。

        精度图的 Input/Output 终端必须用它：用"未被任何 op 产出的候选输入"反推
        会把 TF 的 `.../kernel:0`、converter hoist 常量（`*_hoist_0231`）误判成图输入。
        """
        return {
            'inputs': [t.name() for t in self._ir_graph.get_input_tensors_to_graph()],
            'outputs': [t.name() for t in self._ir_graph.get_output_tensors_of_graph()],
        }


class QnnTruncatedAccuracyAnalysis:
    """截断单算子（truncated layerwise）精度分析：每层从整图切出单算子小图执行。

    语义 = 官方 layerwise：整图其余部分浮点、仅该层量化 -> 误差只反映该层
    自身的量化误差（single_euc / single_cos，与 entire 累积相对）。

    持有 dbg，复用 Golden、后端、工作目录和输入布局；本类负责截断分析与阶段编排。

    每层流程（run() 只做阶段化编排，各阶段私有方法按调用顺序排在下方）：
      1) golden dump（复用 fp32 全图 --debug 结果，缺则补跑一次，不截断 golden）;
      2) 反射量化 DLC 层候选 + 导出 v2 encodings + 模型输入名/raw 映射 + dims;
      3) 逐层准备：subgraph_boundary（目标 DLC op -> 其 ONNX 子图输入边界）
         -> build_cut_onnx + 单层 overrides；无边界或输出不在 ONNX 图的层跳过;
      4) 阶段A converter（进程池：每进程内复用引擎连续转多片，fd 级静默）;
      5) 阶段B tiny DLC meta（qti 主进程直读，每文件新建 reader）;
      6) 阶段C ctx binary（混合精度 DLC 必须离线编译）;
      7) 阶段D 写 fp32 feed + 阶段E net-run --retrieve_context（并行）;
      8) best-match 比对 vs golden（两种布局取最大 cos，量化输出按编码反量化）。

    golden 侧仍是一次 fp32 全图 --debug dump（不截断），供喂数与比对共用。
    切图口径对齐官方 snooping（graph_utils.get_common_parent_activations）：
    目标 DLC op 的子图 = 沿 DLC 图向上回溯到"ONNX 里也存在的张量"作为输入
    边界，因此 converter 融合进该 op 的算子集合被完整包含（例：头部
    Conv+Transpose 融成 1 个 Conv2d），重新转换后算子结构与量化编码一致。
    """

    # converter 进程池复用引擎；补跑每层独立进程，其余阶段使用线程池。
    stage_max_workers = dict(converter=8, converter_retry=8, ctx=8, feed=8, net_run=8)

    # 布局不规范的产出算子：其输出的张量若被当作 cut 图输入，converter 的布局
    # 推断会二次置换（详见 subgraph_boundary 注释），故不能作为子图边界。
    noncanonical_layout_producers = ('Reshape', 'Transpose', 'Flatten', 'Squeeze',
                                     'Unsqueeze', 'Slice', 'Gather', 'Split', 'Expand')

    def __init__(self, dbg: QnnAccuracyDebugger, onnx_path):
        """保存依赖；每轮共享数据由 _collect_run_context 初始化。"""
        self.dbg = dbg
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self._context: dict = {'stage_results': []}

    # ------------------------------------------------------------------
    # 入口：run() —— 阶段化编排（每个阶段委托一个私有方法）
    # ------------------------------------------------------------------

    def run(self, golden_dlc_path: str, target_dlc_path: str, input_list: str,
            only_tensor_prefix=None) -> dict:
        """执行截断单算子逐层分析，返回统一"层精度" dict（single 侧有值、entire 侧占位 None）。

        单层分析不产出 single_mse。这里不做降级：整体失败就抛异常，由调用方
        （QnnAccuracyDebugger.accuracy_analysis）捕获后只输出 entire 报告。
        """
        dbg = self.dbg
        self._context = {'stage_results': []}
        if self.onnx_path is None:
            raise ValueError("QnnTruncatedAccuracyAnalysis needs onnx_path "
                               "(QnnAccuracyDebugger(..., onnx_path=...))")

        # 1) golden dump（复用 dbg.working_dir/golden_output，缺则补跑一次）
        golden_dir = dbg.ensure_golden_dump(golden_dlc_path, input_list)
        golden = dbg.load_dump_dir(golden_dir)
        if not golden:
            raise RuntimeError(f"[QAIRTAccuracyDebugger] golden dir empty: {golden_dir}")
        print(f"[QAIRTAccuracyDebugger] golden layers: {len(golden)}")

        # 2) 层候选 + v2 encodings + 模型输入名/raw 映射 + 全图 NCHW dims
        layers = self._collect_run_context(
            target_dlc_path, input_list, golden, only_tensor_prefix)

        # 3) 准备目标子图与单层 overrides
        infos = self._prepare_all_layers(layers)

        # 4) 阶段A converter（进程池）    5) 阶段B tiny meta（主进程直读）
        converted_infos = self._stage_converter(infos)
        meta_ready_items = self._stage_meta(converted_infos)

        # 6) 阶段C ctx binary    7) 阶段D feed + 阶段E net-run
        ctx_ready_items = self._stage_context_binary(meta_ready_items)
        net_run_ok_items = self._stage_feed_and_netrun(ctx_ready_items)

        # 8) best-match 比对 vs golden
        return self._collect_results(net_run_ok_items, total=len(layers))

    # ------------------------------------------------------------------
    # 阶段私有方法（按 run() 调用顺序排列）
    # ------------------------------------------------------------------

    def _collect_run_context(self, target_dlc_path: str, input_list: str,
                             golden: dict, only_tensor_prefix) -> list:
        """反射层候选（按前缀/golden 过滤）+ 导出 v2 encodings + 输入映射 + 全图 dims。

        模型输入名取 ONNX graph.input 顺序（= prepare_input_data 写 raw 的顺序），
        input_list 首样本按序对应每个输入，多输入模型不写死 input0。
        图索引、编码和输入映射集中保存在本轮上下文，不传入 converter 进程池。
        """
        self._context = ctx = {
            'stage_results': [], 'golden': golden,
            'input_order': str(getattr(self.dbg, 'set_input_order', 'nhwc')).lower(),
            'dlc_ops': {}, 'onnx_producer': {}, 'onnx_inputs': {}, 'onnx_rank': {},
        }
        candidates = self.extract_layer_candidates(target_dlc_path)
        for c in candidates:
            ctx['dlc_ops'].setdefault(c['tensor'], {
                'inputs': c['inputs'], 'op_type': c['op_type']})

        layers = candidates
        if only_tensor_prefix:
            layers = [c for c in layers if c['tensor'].startswith(only_tensor_prefix)]
        layers = [c for c in layers if self._sanitize_name(c['tensor']) in golden]
        print(f"[QAIRTAccuracyDebugger] candidate layers to analyze: {len(layers)}")

        ctx['encodings_path'] = self.extract_dlc_encoding(target_dlc_path)

        full_m = onnx.load(str(self.onnx_path))
        ctx['onnx_model'] = full_m
        ctx['onnx_acts'] = ({o for n in full_m.graph.node for o in n.output}
                            | {i.name for i in full_m.graph.input})
        ctx['model_input_names'] = [i.name for i in full_m.graph.input]
        for n in full_m.graph.node:
            for o in n.output:
                ctx['onnx_producer'][o] = n.op_type
                ctx['onnx_inputs'][o] = list(n.input)
        for v in (list(full_m.graph.value_info) + list(full_m.graph.input)
                  + list(full_m.graph.output)):
            if v.type.tensor_type.HasField('shape'):
                ctx['onnx_rank'][v.name] = len(v.type.tensor_type.shape.dim)
        with open(input_list, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        tokens = first_line.split()
        if len(tokens) != len(ctx['model_input_names']):
            raise RuntimeError(
                f"[QAIRTAccuracyDebugger] input_list raw count ({len(tokens)}) != "
                f"model inputs ({len(ctx['model_input_names'])}): {ctx['model_input_names']}")
        ctx['input_raw_by_name'] = dict(zip(ctx['model_input_names'], tokens))
        ctx['dims_nchw'] = self.onnx_graph_dims_nchw(str(self.onnx_path), model=full_m)
        return layers

    def _prepare_all_layers(self, layers: list) -> list:
        """逐层准备：样本目录只用纯序号 001..，张量名只出现在输出 raw 文件名一次。

        把张量名嵌进绝对路径会超 ~260 字符，深层长名层 qnn-net-run 写文件失败
        （rc17）。一层都准备不出来时直接抛错，不让后续阶段空转。
        """
        dbg = self.dbg
        samples_dir = dbg.working_dir / 'truncated'   # 001/002/... 直接放这里
        samples_dir.mkdir(parents=True, exist_ok=True)

        def prepare(info):
            info['sdir'].mkdir(parents=True, exist_ok=True)
            prepared = self._trunc_prepare_layer(info['sdir'], info)
            if prepared is None:
                return self._stage_result(info, 'prepare', status='skipped',
                                          reason='no ONNX output or usable input boundary')
            return self._stage_result(prepared, 'prepare')

        results = [self._call_stage(prepare, dict(layer, sdir=samples_dir / f'{i + 1:03d}'),
                                   'prepare') for i, layer in enumerate(layers)]
        infos = self._finish_stage(results)
        print(f"[QAIRTAccuracyDebugger] truncated prepared layers: {len(infos)}")
        if not infos:
            raise RuntimeError("[QAIRTAccuracyDebugger] truncated: no layer prepared")
        return infos

    def _stage_converter(self, infos: list) -> list:
        """按完成顺序更新进度，按输入顺序返回成功层。

        进程内复用 converter 引擎；池崩后凭完成标记保留产物，缺失层独立补跑。
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        import tqdm as _tqdm
        # 数字目录会被下一轮复用，不能把旧 DLC 或旧完成标记当成本轮产物。
        ready, cleanup_results = [], []
        for info in infos:
            try:
                out = Path(info['dlc_q'])
                for path in (out, out.with_suffix('.converted'), *(info['sdir'] / name
                             for name in ('comparison.json', 'converter.log', 'converter_retry.log'))):
                    path.unlink(missing_ok=True)
                ready.append(info)
            except OSError as exc:
                cleanup_results.append(self._stage_result(info, 'converter', reason=str(exc)))
        self._finish_stage(cleanup_results)
        infos = ready
        if not infos:
            return []
        jobs = [(str(info['cut_onnx']), str(info['ov_json']), info['layout'],
                 str(info['dlc_q'])) for info in infos]

        results = [None] * len(jobs)
        pool_aborted = False
        try:
            with ProcessPoolExecutor(max_workers=self.stage_max_workers['converter']) as pool:
                idx_of = {pool.submit(QnnSdkTools.convert_one, job): idx
                          for idx, job in enumerate(jobs)}
                with _tqdm.tqdm(total=len(jobs), desc='single[trunc 1/3 converter]',
                                unit='it', ncols=110, file=sys.stderr) as bar:
                    for future in as_completed(idx_of):
                        idx = idx_of[future]
                        try:
                            rc, reason = future.result()
                            results[idx] = self._stage_result(infos[idx], 'converter', rc=rc, reason=reason)
                        except BrokenProcessPool:
                            raise
                        except Exception:
                            results[idx] = self._exception_result(infos[idx], 'converter')
                        bar.update(1)
        except BrokenProcessPool:
            pool_aborted = True

        def _produced(layer_info: dict) -> bool:
            out = Path(layer_info['dlc_q'])
            try:
                return out.is_file() and out.stat().st_size > 0
            except OSError:
                return False

        # rc 未知时还需 worker 的完成标记；非空的半截 DLC 不代表转换成功。
        def _converted(idx: int) -> bool:
            if not _produced(infos[idx]):
                return False
            if results[idx] is None:
                return Path(infos[idx]['dlc_q']).with_suffix('.converted').is_file()
            return results[idx]['status'] == 'ok'

        if pool_aborted:
            lost = [idx for idx in range(len(infos)) if not _converted(idx)]
            print(f"[QAIRTAccuracyDebugger] converter pool aborted; {len(lost)} layer(s) "
                  f"lost, retrying in isolated processes")
            for attempt in (1, 2):
                pending = [idx for idx in lost if not _converted(idx)]
                if not pending:
                    break
                retried = self._run_batch_progress(
                    self._retry_one_isolated, [infos[idx] for idx in pending], 'converter_retry',
                    f'single[trunc 1/3 conv retry {attempt}]',
                    max_workers=self.stage_max_workers['converter_retry'])
                for result in retried:
                    result['attempt'] = attempt
                self._finish_stage(retried)
                for idx, result in zip(pending, retried):
                    results[idx] = dict(result, stage='converter')
                    results[idx].pop('attempt', None)

        for idx, info in enumerate(infos):
            if results[idx] is None:
                results[idx] = self._stage_result(info, 'converter', reason=(
                    '' if _converted(idx) else 'worker exited without a completed DLC'))
            elif results[idx]['status'] == 'ok' and not _produced(info):
                results[idx].update(status='failed', reason='DLC is missing or empty')
        return self._finish_stage(results)

    def _retry_one_isolated(self, info: dict) -> dict:
        """清除残留后独立补跑一层；失败日志落盘，由调用方汇报。"""
        out_path = Path(info['dlc_q'])
        out_path.unlink(missing_ok=True)
        out_path.with_suffix('.converted').unlink(missing_ok=True)
        rc, reason = QnnSdkTools.convert_one_isolated(
            (str(info['cut_onnx']), str(info['ov_json']), info['layout'], str(info['dlc_q'])))
        if rc == 0 and (not out_path.is_file() or not out_path.stat().st_size):
            reason = reason or 'DLC is missing or empty'
        return self._stage_result(info, 'converter_retry', rc=rc, reason=reason)

    def _stage_meta(self, converted_infos: list) -> list:
        """阶段B：读取 tiny meta 写回层任务，剔除读取失败项。"""
        import tqdm

        def read_meta(info):
            info['tiny_meta'] = None
            meta = self.query_tiny_dlc(info['dlc_q'])
            if not meta or not all(meta.get(key) for key in ('inputs', 'outputs', 'tensors')):
                raise ValueError('tiny DLC metadata has no inputs, outputs or tensors')
            for tensor in meta['inputs'] + meta['outputs']:
                if (tensor.get('name') not in meta['tensors'] or tensor.get('dims') is None
                        or any(d <= 0 for d in tensor['dims'])):
                    raise ValueError(f'invalid boundary tensor metadata: {tensor}')
            info['tiny_meta'] = meta
            return self._stage_result(info, 'meta')

        results = [self._call_stage(read_meta, info, 'meta') for info in tqdm.tqdm(
            converted_infos, desc='single[trunc meta]', unit='it', ncols=110,
            file=sys.stderr, disable=not converted_infos)]
        meta_ready_items = self._finish_stage(results)
        print(f"[QAIRTAccuracyDebugger] truncated meta ok: "
              f"{len(meta_ready_items)}/{len(converted_infos)} (in-process)")
        return meta_ready_items

    def _stage_context_binary(self, meta_ready_items: list) -> list:
        """阶段C：ctx binary（并行，子进程静默）。

        混合精度 DLC 含 Convert 节点，qnn-net-run 直接 --dlc_path 会在 composeGraphs
        报 "QNN_DEFINITION_IMPL_GENERATED tensor not supported as output for Convert"，
        必须先离线编译成 context binary 再 --retrieve_context 执行。
        """
        def _ctx(layer_info):
            bin_path = layer_info['sdir'] / 'quant.bin'
            bin_path.unlink(missing_ok=True)
            rc = self.run_qairt_context_binary_generator(
                str(layer_info['dlc_q']), str(bin_path),
                str(layer_info['sdir']), print_output=False)
            reason = 'context binary is missing or empty' if rc == 0 and (
                not bin_path.is_file() or not bin_path.stat().st_size) else ''
            return self._stage_result(layer_info, 'ctx', rc=rc, reason=reason)

        return self._finish_stage(self._run_batch_progress(
            _ctx, meta_ready_items, 'ctx', 'single[trunc 2/3 ctx]',
            max_workers=self.stage_max_workers['ctx']))

    def _stage_feed_and_netrun(self, ctx_ready_items: list) -> list:
        """阶段D 写 fp32 feed + 阶段E net-run --retrieve_context（并行）。

        feed 铁律：输入文件永远是 fp32，net-run 会按该张量编码自行量化定点输入，
        预量化成 uint8 会算出垃圾（详见 _trunc_write_feed）。
        """
        dbg = self.dbg

        # ---- 阶段D：feed（并行）----
        def _feed(layer_info):
            layer_info['input_list'] = None
            path = self._trunc_write_feed(layer_info)
            if not isinstance(path, str) or not Path(path).is_file() or not Path(path).stat().st_size:
                raise ValueError('input list is missing or empty')
            layer_info['input_list'] = path
            return self._stage_result(layer_info, 'feed')

        net_ready_items = self._finish_stage(self._run_batch_progress(
            _feed, ctx_ready_items, 'feed', 'single[trunc feeds]',
            max_workers=self.stage_max_workers['feed']))

        # ---- 阶段E：net-run（并行）----
        def _net(layer_info):
            run_dir = layer_info['sdir'] / 'quant_run'
            if run_dir.exists():
                # 只清本层生成的输出目录；拒绝沿链接删除到其它位置。
                if run_dir.resolve().parent != layer_info['sdir'].resolve():
                    raise ValueError(f'output directory escapes layer directory: {run_dir}')
                shutil.rmtree(run_dir)
            rc = self.run_qnn_net_run_bin(
                str(layer_info['sdir'] / 'quant.bin'), dbg.backend_lib_target,
                layer_info['input_list'], str(run_dir), print_output=False)
            reason = 'no non-empty output in Result_0' if rc == 0 and not any(
                p.stat().st_size for p in (run_dir / 'Result_0').rglob('*.raw')) else ''
            return self._stage_result(layer_info, 'net-run', rc=rc, reason=reason)

        return self._finish_stage(self._run_batch_progress(
            _net, net_ready_items, 'net-run', 'single[trunc 3/3 net-run]',
            max_workers=self.stage_max_workers['net_run']))

    def _collect_results(self, net_run_ok_items: list, total: int) -> dict:
        """阶段8：best-match 比对汇总成统一 dict；一层都没成功时抛错（single 视为失败）。"""
        accuracy = {}
        def compare(info):
            info['comparison'] = self._trunc_compare(info)
            return self._stage_result(info, 'compare', reason=(
                '' if info['comparison'] is not None else 'no finite output with matching dimensions'))

        results = [self._call_stage(compare, info, 'compare') for info in net_run_ok_items]
        for layer_info in self._finish_stage(results):
            compared = layer_info['comparison']
            accuracy[compared['name']] = {
                'entire_cos': None, 'entire_euc': None, 'entire_mse': None,
                'single_cos': compared['cos'], 'single_euc': compared['euc'],
            }
        if not accuracy:
            raise RuntimeError("[QAIRTAccuracyDebugger] truncated: no layer succeeded")
        print(f"[QAIRTAccuracyDebugger] single(truncated) succeeded: "
              f"{len(accuracy)}/{total}")
        return accuracy

    def _stage_result(self, task, stage, *, rc=None, reason='', status=None) -> dict:
        """统一阶段状态；SDK rc 原样保留，异常或无 SDK 调用时为 None。"""
        log_name = {'converter': 'converter.log', 'converter_retry': 'converter_retry.log',
                    'ctx': 'ctx.log', 'net-run': 'quant_run/net-run.log'}.get(stage, f'{stage}.log')
        return dict(task=task, stage=stage, rc=rc, reason=reason,
                    status=status or ('failed' if reason or rc not in (None, 0) else 'ok'),
                    log_path=task['sdir'] / log_name)

    def _exception_result(self, task, stage) -> dict:
        result = self._stage_result(task, stage, reason=traceback.format_exc().strip().splitlines()[-1])
        try:
            with result['log_path'].open('a', encoding='utf-8') as log:
                log.write(traceback.format_exc() + '\n')
        except OSError as exc:
            result['reason'] += f'; log unavailable: {exc}'
        return result

    def _call_stage(self, fn, task, stage) -> dict:
        try:
            return fn(task)
        except Exception:
            return self._exception_result(task, stage)

    def _finish_stage(self, results: list) -> list:
        self._context['stage_results'].extend(results)
        for result in results:
            if result['status'] != 'ok':
                print(f"  [{result['stage']} {result['status']}] {result['task']['tensor']}: "
                      f"rc={result['rc']}; {result['reason']}; log: {result['log_path']}")
        return [result['task'] for result in results if result['status'] == 'ok']

    def save_stage_results(self):
        """保存本轮阶段诊断；写盘失败不影响已有精度结果。"""
        path = self.dbg.working_dir / 'stage_results.json'
        try:
            rows = [dict(tensor=r['task']['tensor'],
                         **{key: str(value) if isinstance(value, Path) else value
                            for key, value in r.items() if key != 'task'})
                    for r in self._context['stage_results']]
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError as exc:
            print(f'[QAIRTAccuracyDebugger] cannot save stage results: {exc}')

    # ------------------------------------------------------------------
    # 单层内部逻辑：准备 / 布局 / 喂数 / 比对（info 字典见 _trunc_prepare_layer）
    # ------------------------------------------------------------------

    def subgraph_boundary(self, target_tensor: str) -> list:
        """目标 DLC op 的 ONNX 子图输入边界（官方 snooping 口径 + 布局规范约束）。

        官方口径：从目标 DLC op 的输入张量出发，张量只要在 ONNX 图里存在就是
        边界；仍是 DLC 内部张量（converter 融合/自造）则沿 DLC 图继续向上。
        因此切出的 ONNX 子图 = 被融合进该 DLC op 的算子集合（例：头部
        Conv+Transpose 融成 1 个 Conv2d），重新转换后算子结构与整图一致。

        额外约束（实测）：边界张量必须是"布局规范"的——产出它的 ONNX 算子不能
        是 Reshape/Transpose 等 shape 算子，且维数应为 4（模型输入除外）。否则
        cut 图把这些张量声明成 NCHW->NHWC 输入时，converter 会二次置换布局：
        - shape 算子产出（如 Transpose 输出）→ Reshape 展平顺序错乱（静默错值）；
        - 3D 张量喂 Concat/Softmax → layout_inferer 直接 IndexError（转换失败）。
        不满足就继续沿 DLC 链（优先）或 ONNX 产出链向上找规范张量。

        Returns: 边界张量名列表（去重）；空 = 该层无可用子图（跳过）。
        """
        ctx = self._context
        boundary: list[str] = []
        seen: set[str] = set()
        stack = list(ctx['dlc_ops'].get(target_tensor, {}).get('inputs', []))
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            prod = ctx['onnx_producer'].get(t)
            rank = ctx['onnx_rank'].get(t)
            if t in ctx['model_input_names'] or (prod is not None
                                           and prod not in self.noncanonical_layout_producers
                                           and (rank is None or rank == 4)):
                boundary.append(t)
            elif t in ctx['dlc_ops']:                     # DLC 自造/融合中间张量
                stack.extend(ctx['dlc_ops'][t]['inputs'])
            elif prod is not None:                       # 非规范 ONNX 中间张量
                stack.extend(ctx['onnx_inputs'].get(t, []))
            # else: 权重/常量等，不构成边界
        return boundary

    def _trunc_prepare_layer(self, sdir: Path, layer: dict) -> dict | None:
        """单层准备：按 DLC-op -> ONNX 子图映射切 cut ONNX + 生成单层 overrides。

        各阶段沿用此字典，后续附加 tiny_meta 与 input_list。
        输出不在 ONNX 图或子图无输入边界（不可截断）时返回 None。
        """
        tensor = layer['tensor']
        if tensor not in self._context['onnx_acts']:
            print(f"[QAIRTAccuracyDebugger] skip (not an onnx tensor): {tensor}")
            return None
        acts = self.subgraph_boundary(tensor)
        if not acts:
            print(f"[QAIRTAccuracyDebugger] skip (no onnx subgraph input): {tensor}")
            return None
        cut_onnx = sdir / 'cut.onnx'
        self.build_cut_onnx(str(self.onnx_path), cut_onnx, acts, tensor,
                            model=self._context['onnx_model'])
        ov_json = sdir / 'overrides.json'
        self.make_layer_override_json(layer['io_tensors'], self._context['encodings_path'],
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
        io = self._context['input_order']
        src_upper = 'NCHW' if str(io).lower() == 'nchw' else 'NHWC'
        model_ins = self._context['model_input_names']
        args = ''
        for a in acts:
            src = src_upper if a in model_ins else 'NCHW'
            args += (f' --source_model_input_layout "{a}" {src}'
                     f' --desired_input_layout "{a}" NHWC')
        return args

    def _trunc_write_feed(self, info: dict) -> str:
        """按 tiny DLC 输入顺序写 fp32 feed，异常由阶段边界统一隔离。"""
        ctx = self._context
        qmeta, golden, dims_nchw = info['tiny_meta'], ctx['golden'], ctx['dims_nchw']
        feed_dir = info['sdir'] / 'feed'
        feed_dir.mkdir(parents=True, exist_ok=True)
        feed_raw_paths = []
        for input_index, tiny_dlc_input in enumerate(qmeta['inputs']):
            input_name, input_dims = tiny_dlc_input['name'], tiny_dlc_input['dims']
            target_dims_nchw = dims_nchw.get(input_name)
            # converter 已声明 desired NHWC；相等维度不能仅凭 shape 猜布局。
            target_layout = self.trunc_order_of(input_dims, target_dims_nchw, preferred_order='nhwc')
            if target_dims_nchw is not None and target_layout is None:
                raise ValueError(f'cannot resolve input layout for {input_name!r}: '
                                 f'{input_dims} vs {target_dims_nchw}')
            if input_name in ctx['model_input_names']:
                feed_array = np.fromfile(ctx['input_raw_by_name'][input_name], dtype=np.float32)
                source_layout = ctx['input_order']
            else:
                # 中间激活直接使用全图 Golden 的 NHWC dump。
                feed_array = golden[self._sanitize_name(input_name)].astype(np.float32)
                source_layout = 'nhwc'
            if target_layout is not None and target_layout != source_layout:
                feed_array = self.trunc_arrange(feed_array, source_layout, target_layout, target_dims_nchw)
            if feed_array.size != int(np.prod(input_dims)) or not np.isfinite(feed_array).all():
                raise ValueError(f'invalid feed for {input_name!r}: {feed_array.size} vs {input_dims}')
            feed_raw_path = feed_dir / f'in{input_index}_{self._sanitize_name(input_name)}.raw'
            feed_array.astype(np.float32).tofile(str(feed_raw_path))
            feed_raw_paths.append(str(feed_raw_path))
        input_list_path = info['sdir'] / 'input_list.txt'
        input_list_path.write_text(' '.join(feed_raw_paths) + '\n', encoding='utf-8')
        return str(input_list_path)

    def _trunc_compare(self, info: dict) -> dict | None:
        """best-match：该层 dump 出的候选数组与 golden 按两种布局比，取 cos 最大者。

        Returns: {'name','cos','euc','mse'}；没有任何尺寸匹配的候选时 None（该层不产出）。
        """
        qmeta = info['tiny_meta']
        golden, dims_nchw = self._context['golden'], self._context['dims_nchw']
        target_tensor = info['tensor']
        golden_tensor_key = self._sanitize_name(target_tensor)
        golden_array = golden[golden_tensor_key]
        if not golden_array.size or not np.isfinite(golden_array).all():
            raise ValueError(f'Golden tensor is empty or non-finite: {target_tensor}')
        target_dims_nchw = dims_nchw.get(target_tensor)
        # 两种布局各算一次 cos：converter 可能给 cut 图输出补 transpose
        golden_references = {'direct': golden_array}
        if target_dims_nchw is not None and len(target_dims_nchw) == 4:
            golden_references['nchw'] = self.trunc_arrange(golden_array, 'nhwc', 'nchw',
                                                           target_dims_nchw)

        def load_dump_array(dump_path: Path):
            """dump 文件 -> fp32 数组。

            位宽按 tiny meta 的 dtype 判定（8/16bit 整型量化输出按其编码反量化）；
            位数与 dtype 不符时（net-run 常把量化张量以 fp32 落盘）退回按 fp32 读。
            """
            dump_bytes = dump_path.stat().st_size
            meta_tensor_name = tensor_names.get(dump_path.stem)
            if meta_tensor_name is None:
                meta_tensor_name = tensor_names.get(dump_path.stem.removesuffix('_native'))
            tensor_entry = qmeta['tensors'].get(meta_tensor_name, {}) if meta_tensor_name else {}
            tensor_dims = tensor_entry.get('dims')
            if tensor_dims is not None and int(np.prod(tensor_dims)) != golden_array.size:
                return None
            numpy_dtype = QnnSdkTools.numpy_dtype_of_qnn_dtype.get(tensor_entry.get('dtype'))
            if (numpy_dtype is not None
                    and dump_bytes == np.dtype(numpy_dtype).itemsize * golden_array.size):
                dump_array = np.fromfile(str(dump_path), dtype=numpy_dtype)
                if np.issubdtype(numpy_dtype, np.floating):
                    return dump_array.astype(np.float32), np.dtype(numpy_dtype).name, meta_tensor_name
                scale = tensor_entry.get('scale')
                if scale is None:
                    return None
                values = dump_array.astype(np.float64)
                scale = np.asarray(scale, dtype=np.float64)
                zero_point = np.asarray(tensor_entry.get('zp', 0), dtype=np.float64)
                if 'axis' in tensor_entry:
                    shape = tensor_entry.get('dims')
                    axis = int(tensor_entry['axis'])
                    if (not shape or not -len(shape) <= axis < len(shape)
                            or int(np.prod(shape)) != values.size):
                        raise ValueError(f'invalid per-axis shape for {meta_tensor_name!r}')
                    axis %= len(shape)
                    broadcast_shape = [1] * len(shape)
                    broadcast_shape[axis] = shape[axis]
                    if scale.size != shape[axis] or zero_point.size not in (1, shape[axis]):
                        raise ValueError(f'invalid per-axis encoding for {meta_tensor_name!r}')
                    values = values.reshape(shape)
                    scale = scale.reshape(broadcast_shape)
                    if zero_point.size != 1:
                        zero_point = zero_point.reshape(broadcast_shape)
                return (((values + zero_point) * scale).astype(np.float32).reshape(-1),
                        np.dtype(numpy_dtype).name, meta_tensor_name)
            if dump_bytes == 4 * golden_array.size:
                return np.fromfile(str(dump_path), dtype=np.float32), 'float32', meta_tensor_name
            return None

        best_cos, best_euc, best_match = -2.0, None, None
        run_dir = info['sdir'] / 'quant_run'
        # 一层只喂一个样本；固定扫描 Result_0，不混入其它样本的输出。
        dump_dir = run_dir / 'Result_0'
        tensor_names = {self._sanitize_name(name): name for name in qmeta['tensors']}
        for dump_path in sorted(dump_dir.rglob('*.raw')):
            loaded = load_dump_array(dump_path)
            if loaded is None:
                continue
            dump_array, dump_dtype, meta_tensor_name = loaded
            if not np.isfinite(dump_array).all():
                continue
            for layout_mode, golden_reference in golden_references.items():
                if dump_array.size != golden_reference.size:
                    continue
                cosine = float(np.dot(golden_reference, dump_array)
                               / (np.linalg.norm(golden_reference) * np.linalg.norm(dump_array) + 1e-12))
                if np.isfinite(cosine) and cosine > best_cos:
                    best_cos = cosine
                    best_euc = float(np.linalg.norm(golden_reference - dump_array))
                    best_match = {'file': str(dump_path.relative_to(run_dir)),
                                  'golden_layout': layout_mode,
                                  'dump_dtype': dump_dtype,
                                  'meta_dtype': qmeta['tensors'].get(meta_tensor_name, {}).get('dtype')}
        if best_cos < -1.5 or not np.isfinite(best_euc):
            print(f"[QAIRTAccuracyDebugger] output missing for {target_tensor}")
            return None
        # 保留现有 best-match 数值策略，逐层记录来源供 SDK/布局差异定位。
        try:
            (info['sdir'] / 'comparison.json').write_text(
                json.dumps({'tensor': target_tensor, 'match': best_match,
                            'cos': best_cos, 'euc': best_euc}, ensure_ascii=False, indent=2),
                encoding='utf-8')
        except OSError as exc:
            print(f"[QAIRTAccuracyDebugger] cannot save comparison for {target_tensor}: {exc}")
        return {'name': golden_tensor_key, 'cos': best_cos, 'euc': best_euc,
                'mse': float(best_euc * best_euc / golden_array.size)}

    # ------------------------------------------------------------------
    # tiny DLC meta（读取本体在 QnnSdkTools；这里只加进度条与落盘）
    # ------------------------------------------------------------------

    def query_tiny_dlc(self, dlc_path) -> dict:
        """读取单个 tiny DLC 图输入顺序/输出/张量量化参数（并落盘 meta json）。"""
        meta = QnnSdkTools.read_tiny_meta(str(dlc_path))
        try:
            QnnSdkTools.write_tiny_meta(str(dlc_path), meta)
        except OSError as exc:
            print(f'[QAIRTAccuracyDebugger] cannot save tiny metadata: {exc}')
        return meta

    # ------------------------------------------------------------------
    # SDK 工具（自带副本；混合精度
    # DLC 须先编译 ctx binary 再 --retrieve_context 执行，原因见上方阶段C）
    # ------------------------------------------------------------------

    def run_qairt_context_binary_generator(self, dlc_path: str, bin_path: str, output_dir: str,
                                           output_tensors: str = None, graph_name: str = None,
                                           print_output: bool = True) -> int:
        """qnn-context-binary-generator 把半量化 DLC 编译成 context binary（.bin）。

        output_tensors 要额外导出的中间张量；--set_output_tensors 需 "graph:tensor"
        语法，故 graph_name 不给时只传张量名。
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
        native_config = getattr(self.dbg, 'native_htp_config', None)
        if native_config:
            cmd += f' --config_file "{native_config}"'
        if output_tensors:
            if graph_name:
                cmd += f' --set_output_tensors "{graph_name}:{output_tensors}"'
            else:
                cmd += f' --set_output_tensors "{output_tensors}"'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]", print_output=print_output,
                           log_path=Path(output_dir) / 'ctx.log')

    def run_qnn_net_run_bin(self, bin_path: str, backend_lib: str, input_list: str,
                            output_dir: str, print_output: bool = True) -> int:
        """执行 qnn-net-run --retrieve_context <bin>（context binary 加载执行）。"""
        cmd = f'qnn-net-run --retrieve_context {bin_path} --backend {backend_lib}'
        cmd += f' --input_list {input_list} --output_dir {output_dir} --log_level error'
        native_config = getattr(self.dbg, 'native_htp_config', None)
        if native_config:
            cmd += f' --config_file "{native_config}"'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]", print_output=print_output,
                           log_path=Path(output_dir) / 'net-run.log')

    def extract_dlc_encoding(self, model_path: str) -> str:
        """量化 DLC -> v2.0.0 encodings json（主进程直读，需 SDK 环境已注入）。"""
        self.dbg.working_dir.mkdir(parents=True, exist_ok=True)
        encoding_path = self.dbg.working_dir / f"{Path(model_path).stem}_encoding.json"
        DlcV2EncodingExporter(model_path).dump(str(encoding_path))
        return str(encoding_path)

    def _sanitize_name(self, name: str) -> str:
        """张量名 -> 输出 raw 文件名（qnn-net-run: 非字母数字下划线统一替换为 '_'）。"""
        return re.sub(r'[^A-Za-z0-9_]', '_', name)

    def extract_layer_candidates(self, quantized_dlc_path: str) -> list[dict]:
        """层候选反射；候选结构与 io_tensors 范围见 DlcV2EncodingExporter.layer_candidates。"""
        return DlcV2EncodingExporter(quantized_dlc_path).layer_candidates()

    def make_layer_override_json(self, io_tensor_names: list, v2_encodings_path: str,
                                 out_json_path: str) -> str:
        """从全量 v2 encodings 筛出该层 io 张量的编码，写成单层 quantization_overrides json。"""
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

    def _run_batch_progress(self, fn, items, stage, desc, *, max_workers: int = 8) -> list:
        """并行执行层任务，按完成顺序推进进度，按输入顺序返回阶段结果。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm

        if not items:
            return []  # 空批次(如前面阶段全部失败)：直接返回，不闪 0 层进度条

        n = len(items)
        results = [None] * n
        # 每批独立持有线程池；不再经全局工具二次读取 Future.result()。
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            idx_of = {pool.submit(self._call_stage, fn, task, stage): i
                      for i, task in enumerate(items)}
            with tqdm.tqdm(total=n, desc=desc, unit='it', ncols=110, file=sys.stderr) as bar:
                for fut in as_completed(idx_of):
                    i = idx_of[fut]
                    results[i] = fut.result()
                    bar.update(1)
        return results

    # ------------------------------------------------------------------
    # 静态纯函数工具（无实例状态；布局重排 / ONNX 切图 / tiny meta 导出）
    # ------------------------------------------------------------------

    @staticmethod
    def onnx_graph_dims_nchw(onnx_path: str, model=None) -> dict:
        """ONNX 内 rank-4 张量的 NCHW dims；传 model 复用已加载的 ModelProto，避免重复 load。"""
        m = model if model is not None else onnx.load(onnx_path)
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
    def trunc_order_of(cls, dims: list, nchw_dims: list | None,
                       preferred_order: str | None = None) -> str | None:
        """按 shape 识别布局；两种 shape 相同只能依赖调用方的显式布局约定。"""
        if nchw_dims is None or len(nchw_dims) != 4 or len(dims) != 4:
            return None
        matches = [order for order, shape in (
            ('nchw', list(nchw_dims)), ('nhwc', cls.trunc_perm_nhwc(nchw_dims)))
            if list(dims) == shape]
        if len(matches) == 1:
            return matches[0]
        return preferred_order if preferred_order in matches else None

    @staticmethod
    def build_cut_onnx(onnx_path: str, out_onnx: Path, act_inputs: list,
                       out_tensor: str, model=None) -> None:
        """从完整 ONNX 切出 [act_inputs -> out_tensor] 子图（张量名保留）。

        model 可传入已加载的 onnx.ModelProto（逐层切图时避免重复 load 大模型）。
        """
        m = model if model is not None else onnx.load(onnx_path)
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


class NetronGraphLayout:
    """按有序内部节点和邻接关系计算 Netron 风格几何，不读取模型或指标。"""

    def __init__(self, node_ids, children: dict, parents: dict, *,
                 node_width: float, node_height: float,
                 node_sep: float, edge_sep: float, rank_sep: float):
        self.nodes = dict.fromkeys(node_ids)
        self.children, self.parents = children, parents
        self.node_width, self.node_height = node_width, node_height
        self.node_sep, self.edge_sep, self.rank_sep = node_sep, edge_sep, rank_sep

    def _layers(self) -> dict[str, int]:
        """dagre 排名：逆拓扑最长路径、network-simplex，再平移并合并空层。"""
        from collections import deque

        layer_row = self.nodes
        children = self.children

        # Kahn 拓扑（源 -> 汇），逆序计算到汇点的最长距离。
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

        rank = self._network_simplex(rank)
        min_rank = min(rank.values(), default=0)
        rank = {n: r - min_rank for n, r in rank.items()}
        present_ranks = sorted(set(rank.values()))
        compact = {r: i for i, r in enumerate(present_ranks)}
        return {n: compact[r] for n, r in rank.items()}

    def _network_simplex(self, rank: dict[str, int]) -> dict[str, int]:
        """dagre networkSimplex：从紧树迭代换出负割边，换入最小 slack 非树边。

        所有边 weight=minlen=1；low/lim 为树的 DFS 区间，用于判断祖先关系。
        原地更新 rank（可为负），由调用方平移并合并空层。
        """
        from collections import defaultdict

        layer_row = self.nodes
        children = self.children
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

        # feasibleTree：扩展紧树，必要时平移树内 rank。
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
                        if v in in_tree:
                            nm = w
                        else:
                            nm = v
                        if nm not in in_tree:
                            in_tree[nm] = True
                            tree_edges.add(frozenset((v, w)))
                            changed = True
            if len(in_tree) >= len(nodes):
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
            if v in in_tree:
                delta = slack((v, w))
            else:
                delta = -slack((v, w))
            for n in list(in_tree):
                rank[n] += delta

        def build_tree_adj():
            adj: dict[str, set[str]] = defaultdict(set)
            for e in tree_edges:
                a, b = tuple(e)
                adj[a].add(b)
                adj[b].add(a)
            return adj

        # 固定树边扫描顺序，防止 PYTHONHASHSEED 改变负割边选择。
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

        def init_cut_values():
            cutvalue: dict[frozenset, float] = {}
            adj = build_tree_adj()
            if not adj:
                return cutvalue
            # 叶子优先（按 pre 从大到小）——树边方向 child->parent
            edge_order = []
            for e in tree_edges_sorted():
                a, b = tuple(e)
                if parent.get(b) == a:
                    child = b
                else:
                    child = a
                if child == b:
                    par = a
                else:
                    par = b
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
                    if pointsToHead:
                        cutvalue[e] += 1.0
                    else:
                        cutvalue[e] += -1.0
                    te = frozenset((child, other))
                    if te in tree_edges:
                        otherCut = cutvalue.get(te, 1.0)
                        if pointsToHead:
                            cutvalue[e] += -otherCut
                        else:
                            cutvalue[e] += otherCut
            return cutvalue

        cutvalue = init_cut_values()

        def leave_edge():
            for e in tree_edges_sorted():
                if cutvalue.get(e, 0) < 0:
                    return e
            return None

        def is_descendant(v_lowlim, root_lowlim):
            root_low, root_lim = root_lowlim
            _, v_lim = v_lowlim
            return root_low <= v_lim <= root_lim

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
            # 从根遍历树，按边方向更新 rank（minlen=1）。
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
        """Netron/dagre 风格 TB 布局：长边虚拟化、交叉最小化、BK 坐标分配。

        返回真实/虚拟节点坐标、每条边的虚拟节点链，以及每层节点数。
        """
        from collections import defaultdict

        node_w, node_h = self.node_width, self.node_height
        node_sep = self.node_sep
        edge_sep = self.edge_sep
        rank_sep = self.rank_sep

        # 固定同 rank 内的初始顺序，避免框架输入行序改变平局结果。
        real = sorted(self.nodes.keys(), reverse=True)
        real_set = set(real)

        # normalize：长边逐层插入虚拟节点，参与相同的排序与定位。
        nodes: dict[str, dict] = {}
        for i, n in enumerate(real):
            nodes[n] = {
                'rank': depth.get(n, 0),
                'width': node_w,
                'height': node_h,
                'dummy': False,
                'seq': i,
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

        # 前驱/后继也按 seq 固定遍历顺序。
        for v in aug_out:
            aug_out[v].sort(key=lambda u: nodes[u].get('seq', 0))
        for v in aug_in:
            aug_in[v].sort(key=lambda u: nodes[u].get('seq', 0))

        layering = self._dagre_order(nodes, aug_in, aug_out)
        x_pos, y_pos = self._dagre_position(
            layering, nodes, aug_in, aug_out, node_sep, edge_sep, rank_sep,
        )

        # 将虚拟节点限制在源/目标横向走廊内；跨度越长，允许的摆幅越大。
        # 只收紧虚拟节点，不改真实节点坐标、order 或 rank。
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
        """dagre.js order（非 compound）：按重心和约束交替扫描，再 transpose。

        down 用前驱、up 用后继；返回按 rank/order 排列的节点层。
        """
        from functools import cmp_to_key

        max_rank = max(nd['rank'] for nd in nodes.values())
        if max_rank < 0:
            return []

        # initOrder：按 rank 升序选起点，后继优先 DFS 首次访问时入层。
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
                    nodes[v]['order'] = i

        assign_order(layering)

        # weight=1，crossCount 等于相邻层边终点的逆序对数。
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

        def barycenter(v: str, relationship: bool) -> dict:
            s = 0.0
            w = 0.0
            if relationship:
                neighbors = aug_in.get(v, [])
            else:
                neighbors = aug_out.get(v, [])
            for u in neighbors:
                o = nodes.get(u, {}).get('order')
                if o is None:
                    continue
                s += o
                w += 1.0
            if w == 0:
                return {'v': v}
            return {'v': v, 'barycenter': s / w, 'weight': w}

        # resolveConflicts：按约束合并会违反已定顺序的节点。
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

            def handle_in(v_entry: dict, u_entry: dict) -> None:
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

            def handle_out(v_entry: dict, w_entry: dict) -> None:
                w_entry['in'].append(v_entry)
                w_entry['indegree'] -= 1
                if w_entry['indegree'] == 0:
                    source_set.append(w_entry)

            while source_set:
                entry = source_set.pop()
                results.append(entry)
                for u in reversed(entry['in']):
                    handle_in(entry, u)
                for w in entry['out']:
                    handle_out(entry, w)

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

        # dagre sort：按重心及 bias 排序，穿插吸收无重心节点。
        def sort(entries: list[dict], bias_right: bool) -> dict:
            lhs = [e for e in entries if 'barycenter' in e]
            rhs = [e for e in entries if 'barycenter' not in e]
            unsortable = sorted(rhs, key=lambda e: -e['i'])

            def compare(a: dict, b: dict) -> int:
                if a['barycenter'] < b['barycenter']:
                    return -1
                if a['barycenter'] > b['barycenter']:
                    return 1
                if bias_right:
                    return b['i'] - a['i']
                else:
                    return a['i'] - b['i']

            sortable = sorted(lhs, key=cmp_to_key(compare))

            def consume_unsortable(vs: list, un: list, index: int) -> int:
                while un and un[-1]['i'] <= index:
                    vs.append(un.pop()['vs'])
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

        def sort_rank(rank_nodes: list[str], cg: dict[str, list[str]], bias_right: bool,
                      relationship: bool) -> list[str]:
            entries = resolve_conflicts([barycenter(v, relationship) for v in rank_nodes], cg)
            return sort(entries, bias_right)['vs']

        # sweepLayerGraphs：逐层扫描，记录层内相邻节点约束。
        def sweep_ranks(ranks_to_sweep: list[int], bias_right: bool, relationship: bool) -> None:
            cg: dict[str, list[str]] = {}
            for r in ranks_to_sweep:
                vs = sort_rank(list(layering[r]), cg, bias_right, relationship)
                layering[r] = vs
                for i, v in enumerate(vs):
                    nodes[v]['order'] = i
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
        for layer in layering:
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
            if vertical == 'u':
                adj = layering
                neighbor = aug_in
            else:
                adj = list(reversed(layering))
                neighbor = aug_out
            for horizontal in ('l', 'r'):
                if horizontal == 'r':
                    adj2 = [list(reversed(layer)) for layer in adj]
                else:
                    adj2 = adj
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
                    if horizontal == 'l':
                        delta = align_range[0] - rng[0]
                    else:
                        delta = align_range[1] - rng[1]
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
                    if vn['dummy']:
                        v_sep = edge_sep
                    else:
                        v_sep = node_sep
                    if un['dummy']:
                        u_sep = edge_sep
                    else:
                        u_sep = node_sep
                    sep = (vn['width'] / 2.0
                           + v_sep / 2.0
                           + u_sep / 2.0
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
            # dagre pass2：包括 dummy 块，向右推至最小出边坐标减间距。
            best = min((xs[c] - s for c, s in block_out[b]), default=None)
            if best is not None:
                xs[b] = max(xs[b], best)

        return {v: xs[root[v]] for layer in layering for v in layer}

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
            if layer:
                last_node = layer[-1]
            else:
                last_node = None
            for i, v in enumerate(layer):
                w = None
                if nodes[v]['dummy']:
                    for u in aug_in.get(v, []):
                        if nodes[u]['dummy']:
                            w = u
                            break
                if w is not None or v == last_node:
                    if w is not None:
                        k1 = nodes[w]['order']
                    else:
                        k1 = prev_len
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
        """dagre.js findType2Conflicts（非 compound），阻止交叉内段垂直对齐。

        扫描相邻层，虚节点的虚前驱超出前后内段边界时标记冲突。
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
        dx, dy = px - x, py - y
        if dx == 0.0 and dy == 0.0:
            return (x, y + node_h / 2.0)
        w, h = node_w / 2.0, node_h / 2.0
        if abs(dy) * w > abs(dx) * h:
            if dy < 0:
                h = -h
            return (x + h * dx / dy, y + h)
        if dx < 0:
            w = -w
        return (x + w, y + w * dy / dx)


    def edge_points(
        self,
        real_pos: dict[str, tuple[float, float]],
        dummy_pos: dict[str, tuple[float, float]],
        edge_routes: dict[tuple[str, str], list[str]],
    ) -> list[tuple[str, str, list[tuple[float, float]]]]:
        """按原邻接顺序返回 (源内部键, 目标内部键, 边界交点及途经点列)。"""
        node_w, node_h = self.node_width, self.node_height
        paths: list[tuple[str, str, list[tuple[float, float]]]] = []

        for a, blist in self.children.items():
            if a not in real_pos:
                continue
            for tb in blist:
                if tb not in real_pos:
                    continue

                route = edge_routes.get((a, tb), [])
                if route:
                    first, last = dummy_pos[route[0]], dummy_pos[route[-1]]
                else:
                    first, last = real_pos[tb], real_pos[a]
                src_pt = self._intersect_rect(real_pos[a], first, node_w, node_h)
                dst_pt = self._intersect_rect(real_pos[tb], last, node_w, node_h)
                if route:
                    pts = [src_pt] + [dummy_pos[d] for d in route] + [dst_pt]
                elif abs(real_pos[tb][0] - real_pos[a][0]) < 1.0:
                    pts = [src_pt, dst_pt]
                else:
                    # 短边控制点向扇出更多的一侧外凸，保持原浮点运算顺序。
                    span = dst_pt[0] - src_pt[0]
                    sgn = -1.0
                    if span > 0:
                        sgn = 1.0
                    if len(self.parents.get(tb, [])) > len(self.children.get(a, [])):
                        sgn = -sgn
                    mx = (src_pt[0] + dst_pt[0]) / 2.0 + sgn * 0.3 * abs(span)
                    my = (src_pt[1] + dst_pt[1]) / 2.0
                    pts = [src_pt, (mx, my), dst_pt]
                paths.append((a, tb, pts))
        return paths

    @staticmethod
    def _netron_curve_path(points: list[tuple[float, float]]) -> str:
        """生成 Netron grapher.Edge.Curve 的漂浮式三次贝塞尔路径。

        第三点先写前导线段，再写曲线；曲线不推进点状态，统一在循环末推进。
        三点及以上补末段曲线，最后直达终点；控制点运算顺序与一位小数格式保持。
        """
        pts = [(float(x), float(y)) for x, y in points]
        n = len(pts)
        if n < 2:
            return ''

        data = ''
        x0 = y0 = x1 = y1 = float('nan')

        def curve(x: float, y: float) -> None:
            nonlocal data
            data += (
                f'C{(2 * x0 + x1) / 3:.1f},{(2 * y0 + y1) / 3:.1f},'
                f'{(x0 + 2 * x1) / 3:.1f},{(y0 + 2 * y1) / 3:.1f},'
                f'{(x0 + 4 * x1 + x) / 6:.1f},{(y0 + 4 * y1 + y) / 6:.1f}'
            )

        for i, (x, y) in enumerate(pts):
            if i == 0:
                data += f'M{x:.1f},{y:.1f}'
            elif i == 2:
                data += f'L{(5 * x0 + x1) / 6:.1f},{(5 * y0 + y1) / 6:.1f}'
                curve(x, y)
            elif i > 2:
                curve(x, y)

            x0, x1 = x1, x
            y0, y1 = y1, y

        if n > 2:
            curve(x1, y1)
        data += f'L{x1:.1f},{y1:.1f}'
        return data


class AccuracyGraph:
    """将 RKNN/QNN 层精度渲染为纵向 DAG 的自包含 HTML。

    rows 保存节点指标，children/parents 保存依赖；累计大字色块与单层小字
    共享可搜索的层列表和详情；几何计算委托 NetronGraphLayout。
    """

    NODE_WIDTH, NODE_HEIGHT = 180.0, 86.0
    METRIC_KEYS = ('entire_cos', 'entire_euc', 'single_cos', 'single_euc', 'entire_mse')

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
        # DLC 反射的 QNN 算子；只补充上面未出现的类型。
        'Conv2d': 'Layer', 'DepthWiseConv2d': 'Layer', 'FullyConnected': 'Layer',
        'TransposeConv2d': 'Layer',
        'Eltwise_Binary': 'Activation', 'Eltwise_Or': 'Activation',
        'Eltwise_And': 'Activation', 'Eltwise_Not': 'Activation',
        'Prelu': 'Activation', 'ReluMinMax': 'Activation',
        'Tanh': 'Activation', 'Gelu': 'Activation', 'HardSwish': 'Activation',
        'ElementWiseNeuron': 'Activation', 'ElementWiseUnary': 'Activation',
        'Resize': 'Transform',   # QNN Resize (最近邻/双线性上采样)
        'PoolMax': 'Pool', 'PoolAvg': 'Pool', 'Pooling': 'Pool',
        'BatchNorm': 'Normalization', 'RmsNorm': 'Normalization',
        'InstanceNorm': 'Normalization',
        "StridedSlice": "Tensor",
    }

    def __init__(
        self,
        accuracy: dict,
        onnx_path: str | Path,
        output_path: str | Path,
        title: str | None = None,
        *,
        sanitize: bool = False,
        include_input: bool = False,
        layer_display=None,
        dlc_candidates: list[dict] | None = None,
        dlc_io: dict | None = None,
    ):
        """由输出张量的精度字典构建报告，ONNX 结构只读取一次。

        accuracy 的值包含 METRIC_KEYS 与可选 op_type；缺失指标保留 None。
        title 默认取结构参考 ONNX 的 stem；output_path 是单 HTML 输出路径。
        sanitize 用于 QNN raw 名称匹配；include_input 追加理想 Input 示意终端。
        layer_display(graph, key) 只还原张量关联键，真实算子名另作界面显示。
        dlc_candidates 非空时按 DLC 的全部候选及实际输入/输出构图，包含融合层；
        否则按 ONNX 张量依赖构图。dlc_io 提供实际图边界，缺失时使用原启发式。
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
        if title is None:
            self.title = self.onnx_path.stem
        self.node_sep = 37.5 # 横向：真实节点之间的间距
        self.edge_sep = 15.0 # 横向：长边虚拟节点占用的间距
        self.rank_sep = 37.5 # 纵向：层与层之间的间距
        self.pan_speed:float = 15

        self._parse_accuracy_dict()

    # ------------------------------------------------------------------
    # 内部：由统一 dict 构建 onnx 图 / 层图 / 终端节点
    # ------------------------------------------------------------------

    def _parse_accuracy_dict(self) -> None:
        """按当前图视角组装精度行、邻接关系和示意终端。"""
        if self._dlc_candidates:
            self._parse_accuracy_from_dlc()
        else:
            self._parse_accuracy_from_onnx()

    def _finish_parse(self, real_rows: list[dict], children: dict, parents: dict,
                      input_rows: list[dict], output_rows: list[dict],
                      input_layers: list[str], output_layers: list[str]) -> None:
        """收尾：组装 self.rows / children / parents / 颜色范围（两模式共用）。"""
        for row in input_rows + output_rows:
            row['terminal'] = True
        self.rows = input_rows + real_rows + output_rows
        self.inputs = input_layers
        self.outputs = output_layers
        self.children = children
        self.parents = parents
        self.data = {'rows': self.rows, 'inputs': self.inputs, 'outputs': self.outputs}
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
        # layer_display 还原张量关联键；算子展示名另存，避免重名/多输出合并。
        name_of = lambda key: key
        if self._layer_display is not None:
            name_of = lambda key: self._layer_display(graph, key)

        results, real_rows = [], []
        for name, row in self._accuracy.items():
            raw_stem = str(name).removesuffix('.raw')
            tensor = match_tensor(raw_stem, graph)
            exact = tensor is not None and raw_stem in {tensor, '_' + tensor, tensor.lstrip('_')}
            # op_type：优先 dict 自带（Rknn error_analysis），缺则 onnx node_info 推断
            op_type = row.get('op_type')
            if not op_type and tensor is not None and tensor in node_info:
                op_type = node_info[tensor][0]
            display = name_of(raw_stem)
            label, tensor_name = display, raw_stem
            if exact:
                label = graph['node_labels'].get(tensor) or display
                tensor_name = graph['tensors'].get(tensor, raw_stem)
            parsed = {
                'layer_name': display,
                'op_type': op_type or 'Unknown',
                'display_name': label, 'tensor_name': tensor_name,
                **{key: row.get(key) for key in self.METRIC_KEYS},
            }
            real_rows.append(parsed)
            results.append(dict(parsed, tensor=tensor))

        children, parents, _ = build_layer_graph(results, graph)

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
        """按实际 DLC 候选构图，张量关联键与算子展示名保持独立。"""
        graph = build_onnx_tensor_graph(self.onnx_path, sanitize=self._sanitize)
        name_of = lambda key: key
        if self._layer_display is not None:
            name_of = lambda key: self._layer_display(graph, key)

        cands = self._dlc_candidates

        # 每个输出保留一个分析项，精度缺失也保留；重复张量取首条。
        produced, real_rows = set(), []
        for c in cands:
            t = c['tensor']
            if t in produced:
                continue
            produced.add(t)
            acc = self._accuracy.get(re.sub(r'[^A-Za-z0-9_]', '_', t)) or {}
            label, outputs = c.get('op_name') or '', c.get('outputs', [t])
            if label and len(outputs) > 1 and t in outputs:
                label += f' [输出 {outputs.index(t)}]'
            display = name_of(t)
            real_rows.append({
                'layer_name': display, 'display_name': label or display, 'tensor_name': t,
                'op_type': c['op_type'],
                **{key: acc.get(key) for key in self.METRIC_KEYS},
            })

        # 生产张量连到消费它的候选；兼容旧候选的 io_tensors 格式。
        children: dict[str, list[str]] = defaultdict(list)
        parents: dict[str, list[str]] = defaultdict(list)

        for c in cands:
            cur = c['tensor']
            ins = c.get('inputs', c['io_tensors'][:-1])
            for i in ins:
                if i in produced and i != cur:
                    children[i].append(cur)
                    parents[cur].append(i)
        for adjacency in (children, parents):
            for key, values in adjacency.items():
                adjacency[key] = list(dict.fromkeys(values))

        # 边键统一到关联键；不得改用可能重名的算子展示名。
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

        def consumers(t: str) -> list[str]:
            # 一个图输入可被多个 op 或同一 op 的多个输出候选消费。
            return list(dict.fromkeys(
                display_of.get(c['tensor'], c['tensor']) for c in cands
                if t in c.get('inputs', c['io_tensors'][:-1])))

        input_rows: list[dict] = []
        input_layers: list[str] = []
        output_rows: list[dict] = []
        output_layers: list[str] = []
        if self._include_input:
            # 优先 IR 边界；缺失时启发式可能把 TF kernel/hoist 常量误判为输入。
            if self._dlc_io and self._dlc_io.get('inputs'):
                model_ins = [t for t in self._dlc_io['inputs'] if t not in produced]
            else:
                all_ins: set[str] = set()
                for c in cands:
                    all_ins.update(c.get('inputs', c['io_tensors'][:-1]))
                model_ins = sorted(
                    t for t in (all_ins - produced)
                    if not any(k in t for k in ('weight', 'bias', 'coeff', 'onnx::')))
            for inp in model_ins:
                display = inp  # 图输入没有 DLC 节点，直接显示原名
                input_layers.append(display)
                input_rows.append({
                    'layer_name': display, 'op_type': 'Input',
                    'entire_cos': 1.0, 'entire_euc': 0.0,
                    'single_cos': 1.0, 'single_euc': 0.0,
                })
                for con in consumers(inp):
                    children_disp.setdefault(display, []).append(con)
                    parents_disp.setdefault(con, []).append(display)

        # 无 IR 输出边界时，以没有下游消费者的候选作为输出。
        out_set = None
        if self._dlc_io and self._dlc_io.get('outputs'):
            out_set = set(self._dlc_io['outputs'])
        row_by_name = {}
        for row in real_rows:
            row_by_name.setdefault(row['layer_name'], row)
        for c in cands:
            t = c['tensor']
            if out_set is not None:
                if t not in out_set:      # 非 IR 图输出 -> 不画 Output 终端
                    continue
            elif t in children:           # 启发式：有下游消费者 -> 非图输出
                continue
            display = display_of.get(t, t)
            node_name = display
            if display in row_by_name:
                node_name += ' (out)'
            output_layers.append(node_name)
            row = row_by_name.get(display, {})
            output_rows.append({
                'layer_name': node_name, 'op_type': 'Output',
                **{key: row.get(key) for key in
                   self.METRIC_KEYS[:4]},
            })
            children_disp.setdefault(display, []).append(node_name)
            parents_disp.setdefault(node_name, []).append(display)

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
        category = self.OP_CATEGORY.get(op_type or '', '')
        return self.CATEGORY_COLORS.get(category, self.DEFAULT_COLOR)

    def _euc_to_color(self, euc: float | None) -> str:
        """欧氏距离映射为绿→黄→红，范围中点起为纯红（cos≤0.8）。"""
        if euc is None:
            return '#b0b0b0'
        span = self.euc_color_max - self.euc_color_min
        if span <= 0:
            t = 0.0
        else:
            t = (euc - self.euc_color_min) / span
        return self._cos_to_color(1.0 - max(0.0, min(1.0, t)) * 0.4)

    @staticmethod
    def _text_color_for(bg: str) -> str:
        """浅背景用灰黑文字，深背景用白色（BT.709 相对亮度）。"""
        bg = bg.lstrip('#')
        if len(bg) != 6:
            return '#fff'
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
        # 相对亮度 (ITU-R BT.709)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum > 140:
            return '#333333'
        return '#fff'

    def _layout_engine(self) -> NetronGraphLayout:
        return NetronGraphLayout(self.layer_row, self.children, self.parents,
                                 node_width=self.NODE_WIDTH, node_height=self.NODE_HEIGHT,
                                 node_sep=self.node_sep, edge_sep=self.edge_sep, rank_sep=self.rank_sep)

    def _layers(self) -> dict[str, int]:
        return self._layout_engine()._layers()

    def _compute_layout(self, depth: dict[str, int]):
        return self._layout_engine()._compute_layout(depth)

    _intersect_rect = staticmethod(NetronGraphLayout._intersect_rect)
    _netron_curve_path = staticmethod(NetronGraphLayout._netron_curve_path)

    # ------------------------------------------------------------------
    # 构建与渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt5(value: float | None) -> str:
        """五位有效数字，整数和科学计数法尾数至少保留一位小数。"""
        if value is None:
            return '—'
        mantissa, separator, exponent = f'{value:.5g}'.partition('e')
        if '.' not in mantissa:
            mantissa += '.0'
        return mantissa + separator + exponent

    def _svg_node(self, name: str, op: str, row: dict) -> tuple[str, str]:
        """生成类型色栏、累计大字色块和单层小字组成的节点。"""
        import base64

        width, height = int(self.NODE_WIDTH), int(self.NODE_HEIGHT)
        top_fill = self._op_color(op)
        top_text = self._text_color_for(top_fill)
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<defs><clipPath id="cp"><rect x="0" y="0" width="{width}" height="{height}" '
            'rx="12" ry="12"/></clipPath></defs>',
            f'<rect x="0" y="0" width="{width}" height="34" fill="{top_fill}" clip-path="url(#cp)"/>',
            f'<text x="{width/2}" y="24" font-family="Arial" font-size="16" fill="{top_text}" '
            f'text-anchor="middle" font-weight="bold">{html.escape(op)}</text>',
            f'<rect x="0" y="34" width="{width}" height="52" fill="#fff" clip-path="url(#cp)"/>',
            f'<line x1="{width//2}" y1="39" x2="{width//2}" y2="81" stroke="#edf1f5"/>',
        ]
        for metric, left, right, color in [('cos', 8, 82, self._cos_to_color(row.get('entire_cos'))),
                                           ('euc', 98, 172, self._euc_to_color(row.get('entire_euc')))]:
            svg.extend([
                f'<text x="{left}" y="44" font-family="Arial" font-size="8.5" '
                f'fill="#7c8b9d">entire {metric}</text>',
                f'<rect class="accuracy-color" data-key="entire_{metric}" '
                f'x="{left-3}" y="48" width="80" height="21" rx="3" fill="{color}"/>',
                f'<text x="{left}" y="79" font-family="Arial" font-size="8.5" '
                'fill="#7c8b9d">single</text>',
            ])
            for stage, y, size, limit, small, weight, fill, missing in [
                    ('entire', 64, 17, 8, 12, '600', '#17251f', '#34465c'),
                    ('single', 79, 9.5, 9, 8, '400', '#34465c', '#9aa5b4')]:
                key = f'{stage}_{metric}'
                value = row.get(key)
                label = self._fmt5(value)
                if value is None:
                    fill = missing
                if len(label) > limit:
                    size = small
                svg.append(f'<text class="accuracy-value" data-key="{key}" x="{right}" y="{y}" '
                           f'text-anchor="end" font-family="Consolas,monospace" font-size="{size}" '
                           f'font-weight="{weight}" fill="{fill}">{html.escape(label)}</text>')
        svg.append(f'<rect x="0.5" y="0.5" width="{width-1}" height="{height-1}" rx="12" ry="12" '
                   'fill="none" stroke="#94a3b8" stroke-width="1"/></svg>')
        image = 'data:image/svg+xml;base64,' + base64.b64encode(''.join(svg).encode('utf-8')).decode('ascii')
        title = '\n'.join([name, f'op_type: {op}'] + [f'{key}={row.get(key)}' for key in
                         ('entire_cos', 'entire_euc', 'single_cos', 'single_euc')])
        return image, title


    def _build_edge_paths(self, real_pos: dict, dummy_pos: dict, edge_routes: dict, *,
                          layout: NetronGraphLayout | None = None) -> list:
        """给有序几何点列附加精度颜色和原 tooltip，保留旧私有返回契约。"""
        if layout is None:
            layout = self._layout_engine()
        paths = []
        for source, target, points in layout.edge_points(real_pos, dummy_pos, edge_routes):
            color = self._cos_to_color(self.layer_row[source].get('entire_cos'))
            value = self.layer_row[target].get('entire_cos')
            title = f'{source} -> {target}'
            if value is not None:
                title += f'\nentire_cos={value:.6f}'
            paths.append((points, color, title))
        return paths

    _REPORT_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;color:#233047;background:#f5f7fa;font-synthesis:none;--line:#e3e8ee;--muted:#7c8797;--accent:#087b75}
*{box-sizing:border-box}body{margin:0;height:100vh;overflow:hidden}button,input,select{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #0c9489;outline-offset:3px}button{color:inherit}a{color:inherit;text-decoration:none}
.app{--layers-width:252px;--details-width:300px;position:relative;height:100%;display:grid;grid-template-rows:minmax(0,1fr)}.plain{border:1px solid var(--line);border-radius:6px;background:#fff;font-size:12px;padding:7px 11px;display:inline-flex;align-items:center;justify-content:center;gap:6px}.plain:hover{background:#f4f7f9}.plain svg{width:15px;height:15px}
.workspace{display:grid;grid-template-columns:var(--layers-width) minmax(0,1fr) var(--details-width);grid-template-rows:minmax(0,1fr);min-height:0}.sidebar,.inspector{background:#fff;min-height:0;display:flex;flex-direction:column;z-index:3}.sidebar{grid-column:1;border-right:1px solid var(--line)}.inspector{grid-column:3;border-left:1px solid var(--line);overflow:auto}.side-top{padding:19px 16px 14px}.section-title{display:flex;align-items:center;justify-content:space-between;height:18px;padding-left:80px;gap:6px;white-space:nowrap;font-size:13px;font-weight:650}.count{flex-shrink:0;font-size:10px;background:#f0f3f7;color:#64748b;border-radius:4px;padding:2px 4px;font-weight:500}.search{position:relative;margin-top:15px}.search svg{width:15px;height:15px;position:absolute;left:10px;top:11px;color:#95a1b0}.search input{width:100%;height:36px;padding:0 10px 0 32px;border:1px solid #dde4eb;border-radius:6px;background:#fafbfd;font-size:12px;color:#28394c}.search input::placeholder{color:#99a3b0}.sort-row{display:flex;align-items:center;justify-content:space-between;padding:0 16px 12px;font-size:10px;color:#99a3b0}.sort-row select{border:0;background:transparent;color:#778499;max-width:140px;font-size:10px;cursor:pointer}.layer-list{overflow:auto;flex:1;padding:0 9px 12px}.layer{border:1px solid transparent;background:#fff;width:100%;text-align:left;border-radius:7px;padding:10px 9px;margin-bottom:3px;display:grid;grid-template-columns:21px 1fr;gap:6px}.layer:hover{background:#f7f9fb}.layer.selected{border-color:#bfdbd6;background:#edf5f3}.layer-index{font:10px ui-monospace,Consolas,monospace;color:#a2acb8;padding-top:2px}.layer-title{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:168px}.layer-op{font-size:10px;color:#95a0af;margin-top:4px}.layer-bottom{display:flex;gap:9px;margin-top:9px;font:10px ui-monospace,Consolas,monospace;color:#7d8a9d}.layer-bottom .label{font-size:9px;color:#a0a9b5;margin-right:3px}.dot{width:5px;height:5px;display:inline-block;border-radius:50%;background:#66a78c;vertical-align:middle;margin-right:4px}.dot.missing{background:#b8c0ca}.empty{padding:25px 8px;color:#8b96a5;font-size:12px;text-align:center}.side-foot{border-top:1px solid var(--line);font-size:10px;line-height:1.7;color:#98a3b2;padding:12px 17px}.coverage{height:3px;background:#edf1f5;border-radius:2px;margin-top:7px;overflow:hidden}.coverage i{display:block;height:100%;background:#8ab6a9}
.canvas{grid-column:2;min-width:0;position:relative;overflow:hidden;background:#f5f7fa}.canvas>svg{width:100%;height:100%;display:block;touch-action:none;user-select:none;cursor:grab}.canvas>svg image.node{cursor:pointer}.canvas-empty{position:absolute;left:0;right:0;top:45%;text-align:center;font-size:13px;color:#8d99aa;pointer-events:none}.canvas-caption{min-width:0;font-size:10px;color:#94a0b1}.canvas-caption h1{margin:0 0 6px;color:#34465c;font-size:14px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.view-controls{position:absolute;left:18px;bottom:18px;display:flex;gap:0;border:1px solid #dce3ea;border-radius:7px;box-shadow:0 3px 14px #21334c08;background:#fff;padding:3px;align-items:center}.view-controls button{width:31px;height:29px;background:#fff;border:0;font-size:17px;border-radius:4px;display:grid;place-items:center}.view-controls button:hover{background:#eef3f5}.view-controls output{min-width:47px;text-align:center;color:#68778a;font:10px ui-monospace,Consolas,monospace}.view-controls .separator{height:15px;width:1px;background:#e4e9ef;margin:0 4px}.view-controls svg{width:15px;height:15px}.map{position:absolute;bottom:18px;right:18px;width:110px;height:220px;border:1px solid #dce3ea;border-radius:7px;background:#fffffff2;box-shadow:0 3px 14px #21334c05;padding:7px;cursor:crosshair;touch-action:none;user-select:none}.map.dragging{cursor:grabbing}.map svg{display:block;width:100%;height:100%;overflow:hidden}.map rect.viewport{fill:#197e7510;stroke:#459388;stroke-width:1.3;vector-effect:non-scaling-stroke}.canvas-key{position:absolute;bottom:64px;left:18px;color:#8d99aa;font-size:10px;pointer-events:none}
.inspect-header{padding:20px 19px 17px;border-bottom:1px solid var(--line)}.inspect-kicker{font-size:12px;line-height:24px;color:#7e8a9c;margin-bottom:12px}.inspect-heading{display:flex;align-items:center;justify-content:space-between;gap:8px}.op-badge{border-radius:4px;background:#edf1f8;color:#577193;padding:4px 7px;font-size:10px;font-weight:650}.copy{border:0;background:transparent;color:#97a3b3;padding:4px;display:grid;place-items:center}.copy svg{width:15px;height:15px}.full-name{font:12px/1.65 ui-monospace,Consolas,"Microsoft YaHei",monospace;word-break:break-all;margin-top:13px;color:#34465c}.inspect-section{padding:19px;border-bottom:1px solid var(--line)}.small-title{margin:0 0 12px;color:#7e8a9c;font-size:10px;font-weight:600;letter-spacing:.5px}.metrics{width:100%;border-collapse:collapse;font-size:11px}.metrics th{color:#96a1b1;font-size:10px;font-weight:400;padding:0 0 11px;text-align:right}.metrics th:first-child,.metrics td:first-child{text-align:left}.metrics td{padding:10px 0;border-top:1px solid #f0f3f6;text-align:right;font:11px ui-monospace,Consolas,monospace;color:#506176}.metrics td:first-child{font-family:inherit;font-size:11px;color:#758297}.metrics td.missing{font-size:10px;color:#a6afbb}.relation{display:block;text-align:left;border:1px solid #e8edf2;width:100%;background:#fbfcfd;border-radius:5px;color:#69788b;padding:8px;margin:6px 0;font:10px/1.6 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}.relation:hover{background:#eff5f3;border-color:#c9ddd8}.relation-head{display:flex;justify-content:space-between;font-size:10px;color:#9aa4b3;margin-top:11px}.legend-scale+.legend-scale{margin-top:18px}.legend-heading{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#68788d}.legend-heading small{font-size:9px;color:#98a3b1}.legend-bar{height:7px;border-radius:5px;background:linear-gradient(to right,#ff4500 0%,#ffd700 50%,#3cb371 100%);margin-top:9px}.legend-euc .legend-bar{background:linear-gradient(to right,#3cb371 0%,#ffd700 25%,#ff4500 50%,#ff4500 100%)}
.legend-numbers{position:relative;height:14px;color:#8d99aa;font:9px ui-monospace,Consolas,monospace;margin-top:6px}.legend-numbers span{position:absolute;top:0;white-space:nowrap;transform:translateX(-50%)}.legend-numbers span:first-child{transform:none}.legend-numbers span:last-child{transform:translateX(-100%)}.legend-scale.single-value .legend-numbers span{left:0!important;transform:none}.legend-text{font-size:10px;line-height:1.85;color:#8d99aa;margin:8px 0 0}
.legend-key{display:flex;gap:8px;align-items:center;color:#8795a8;font-size:10px;margin-top:12px}.legend-node{display:inline-flex;overflow:hidden;border-radius:3px;border-top:3px solid #577193}.legend-node i{font:8px/15px ui-monospace,Consolas,monospace;padding:0 5px;background:#3cb371;color:#17251f}.legend-node i+i{background:#ffd700}.legend-missing{width:12px;height:12px;background:#b0b0b0;border-radius:3px}
.tensor-name{margin-top:8px;font:10px/1.6 ui-monospace,Consolas,monospace;color:#8d99aa;overflow-wrap:anywhere}
.tensor-name:not([hidden]){display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px}.tensor-name>span:first-child{white-space:nowrap}
.toast{position:fixed;left:50%;bottom:53px;transform:translateX(-50%);background:#213a43;color:white;font-size:11px;padding:8px 14px;border-radius:6px;display:none;z-index:20}.panel-btn{position:absolute;top:16px;z-index:5}.layers-btn{left:16px}.app.hide-layers .canvas-head{left:104px}.details-btn{right:16px}.app.hide-layers{--layers-width:0px}.app.hide-details{--details-width:0px}.app.hide-layers .sidebar,.app.hide-details .inspector{display:none}
@media(max-width:1150px){.app{--layers-width:222px;--details-width:272px}.layer-title{max-width:144px}}
@media(max-width:900px){.app{--layers-width:210px;--details-width:0px}.inspector{position:absolute;top:0;right:0;bottom:0;width:272px;box-shadow:-10px 0 40px #1c31400d}.layer-title{max-width:132px}}
@media(max-width:600px){.app{--layers-width:0px}.sidebar{position:absolute;top:0;left:0;bottom:0;width:252px;box-shadow:10px 0 40px #1c31400d}.layer-title{max-width:168px}.canvas-caption h1{font-size:12px}.map{width:90px;height:180px}}

.canvas-head{position:absolute;top:18px;left:19px;right:18px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,42%);gap:18px;pointer-events:none;z-index:2}
.canvas-status{min-width:0;display:flex;flex-direction:column;gap:6px;text-align:right;color:#8d99aa;font-size:10px;line-height:1.5}.canvas-status>span{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.app.hide-details .canvas-head{right:104px}.canvas-key{display:flex;flex-direction:column;gap:5px;line-height:1.5}.canvas-key kbd{border:1px solid #dce3ea;border-radius:3px;padding:1px 4px;font:9px ui-monospace,Consolas,monospace;background:#f9fafb}
</style>
</head>
<body>
<div class="app" id="app">
  <button class="plain panel-btn layers-btn" id="toggle-layers" aria-label="显示或隐藏层列表" aria-expanded="true" aria-controls="layer-sidebar"><span id="layers-label">收起列表</span></button>
  <button class="plain panel-btn details-btn" id="toggle-details" aria-label="显示或隐藏层详情" aria-expanded="true" aria-controls="layer-inspector"><span id="details-label">收起详情</span></button>
  <div class="workspace">
    <aside class="sidebar" id="layer-sidebar" aria-label="层列表">
      <div class="side-top"><div class="section-title"><span id="layers-heading">层目录</span><span class="count" id="layer-count"></span></div>
        <div class="search"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8.5" cy="8.5" r="5.5"/><path d="m13 13 4 4"/></svg><input id="search" type="search" autocomplete="off" aria-label="搜索层名、张量或算子" placeholder="搜索层名、张量或算子…"></div>
      </div>
      <div class="sort-row"><span>S 单层 · E 累计</span><select id="sort" aria-label="层列表排序"><option value="order">计算图顺序</option><option value="entire_cos">累计 cos 升序</option><option value="single_cos">单层 cos 升序</option></select></div>
      <div class="layer-list" id="layer-list"></div>
      <div class="side-foot"><span id="coverage"></span><div class="coverage"><i id="coverage-bar"></i></div></div>
    </aside>
    <main class="canvas" aria-label="计算图画布">
      <div class="canvas-head"><div class="canvas-caption"><h1 id="model-title"></h1><span id="source-note">点击节点查看完整指标</span></div><div class="canvas-status"><span id="graph-count"></span><span id="selection-status"></span></div></div><div class="canvas-key"><span><kbd>W A S D</kbd> 平移画布</span><span>滚轮缩放 · 拖拽平移</span></div>
      __SVG__<div class="canvas-empty" id="canvas-empty" hidden>暂无计算图节点</div>
      <div class="view-controls"><button id="zoom-out" aria-label="缩小">−</button><output id="zoom-value">100%</output><button id="zoom-in" aria-label="放大">+</button><span class="separator"></span><button id="fit" aria-label="还原视图"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4"/></svg></button></div>
      <div class="map" id="map" title="点击定位，拖动平移画布"><svg id="map-svg" aria-label="全图缩略图"></svg></div>
    </main>
    <aside class="inspector" id="layer-inspector" aria-label="选中层详情">
      <div class="inspect-header"><div class="inspect-kicker">层详情</div><div class="inspect-heading"><span class="op-badge" id="detail-op"></span><button class="copy" id="copy-name" aria-label="复制完整层名" title="复制完整层名"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="7" y="7" width="10" height="11" rx="1.5"/><path d="M13 7V3H3v11h4"/></svg></button></div><div class="full-name" id="detail-name"></div><div class="tensor-name" id="detail-tensor" hidden><span>输出张量：</span><span id="detail-tensor-value"></span></div></div>
      <div class="inspect-section"><h2 class="small-title">精度指标</h2><table class="metrics"><thead><tr><th>指标</th><th>单层 single</th><th>累计 entire</th></tr></thead><tbody id="metric-rows"></tbody></table></div>
      <div class="inspect-section"><h2 class="small-title">相邻层</h2><div id="relations"></div></div>
      <div class="inspect-section"><h2 class="small-title">图例</h2>
        <div class="legend-scale" id="cos-legend"><div class="legend-heading"><span>累计 cos</span><small>越大越接近参考值</small></div><div class="legend-bar"></div><div class="legend-numbers"><span style="left:0">≤0.80</span><span style="left:50%">0.90</span><span style="left:100%">1.00</span></div></div>
        <div class="legend-scale legend-euc" id="euc-legend"><div class="legend-heading"><span>累计 euc</span><small>本图范围 · 越小越接近参考值</small></div><div class="legend-bar"></div><div class="legend-numbers" id="euc-ticks"></div></div>
        <div class="legend-key"><span class="legend-node"><i>cos</i><i>euc</i></span><span>累计：大字色块 · 单层：下方小字</span></div>
        <div class="legend-key"><span class="legend-missing"></span><span>未提供</span></div>
      </div>
    </aside>
  </div>
</div>
<div class="toast" id="toast" role="status"></div>
<script>
const payload = __PAYLOAD__;
const rows=payload.rows, layers=rows.filter(row=>!row.terminal), svg=document.getElementById('svg');
const $=id=>document.getElementById(id), ns=svg.namespaceURI;
let selected=layers[0]||rows[0], baseView;
const number=value=>value==null?'未提供':Number(value).toPrecision(6);
const short=value=>value.split('/').filter(Boolean).slice(-2).join(' / ');
const layerName=row=>row?.display_name||row?.layer_name||'';
const byName=new Map(rows.map(row=>[row.layer_name,row]));
function text(tag,value,cls){const element=document.createElement(tag);element.textContent=value;if(cls)element.className=cls;return element;}
function compact(value){const [mantissa,exponent]=Number(value.toPrecision(4)).toString().split('e');return mantissa+(mantissa.includes('.')?'':'.0')+(exponent===undefined?'':`e${exponent}`);}
const view=()=>svg.getAttribute('viewBox').split(/[ ,]+/).map(Number);
function setView(next){svg.setAttribute('viewBox',next.join(' '));syncView();}
function syncView(){
  const current=view(),box=$('map-svg').querySelector('.viewport');
  $('zoom-value').textContent=`${Math.round(baseView[2]/current[2]*100)}%`;
  if(box)['x','y','width','height'].forEach((key,index)=>box.setAttribute(key,current[index]));
}
function point(element,event){return new DOMPoint(event.clientX,event.clientY).matrixTransform(element.getScreenCTM().inverse());}
function zoomAt(x,y,factor){const current=view();setView([x+(current[0]-x)*factor,y+(current[1]-y)*factor,current[2]*factor,current[3]*factor]);}
function renderList(){
  const query=$('search').value.toLowerCase().trim(),sort=$('sort').value;
  let list=layers.filter(row=>!query||`${layerName(row)} ${row.tensor_name||''} ${row.layer_name} ${row.op_type}`.toLowerCase().includes(query));
  if(sort!=='order')list=list.slice().sort((a,b)=>(a[sort]??Infinity)-(b[sort]??Infinity));
  $('layer-list').replaceChildren();$('layer-count').textContent=`${list.length} / ${layers.length}`;
  for(const row of list){
    const button=text('button','','layer'+(row===selected?' selected':''));button.title=layerName(row)+(row.tensor_name?`\n输出张量：${row.tensor_name}`:'');
    button.append(text('span',String(layers.indexOf(row)+1).padStart(2,'0'),'layer-index'));
    const content=text('div','');content.append(text('div',short(layerName(row)),'layer-title'),text('div',row.op_type,'layer-op'));
    const metrics=text('div','','layer-bottom');
    for(const [label,value] of [['S',row.single_cos],['E',row.entire_cos]]){
      const item=text('span','');item.append(text('i','','dot'+(value==null?' missing':'')),text('span',label,'label'),text('span',value==null?'—':value.toFixed(4)));metrics.append(item);
    }
    content.append(metrics);button.append(content);button.onclick=()=>choose(row,true);$('layer-list').append(button);
  }
  if(!list.length)$('layer-list').append(text('div',layers.length?'没有匹配的层':'暂无分析层','empty'));
}
function choose(row,focus=false){
  if(!row)return;selected=row;renderList();$('detail-name').textContent=layerName(row);$('detail-op').textContent=row.op_type;
  $('detail-tensor').hidden=!row.tensor_name||row.tensor_name===layerName(row);$('detail-tensor-value').textContent=row.tensor_name||'';
  $('selection-status').textContent=`已选中 ${short(layerName(row))}`;$('metric-rows').replaceChildren();
  for(const [label,single,entire] of [['Cosine','single_cos','entire_cos'],['Euclidean','single_euc','entire_euc']]){
    const tr=document.createElement('tr');tr.append(text('td',label));
    for(const key of [single,entire])tr.append(text('td',number(row[key]),row[key]==null?'missing':''));$('metric-rows').append(tr);
  }
  if(row.entire_mse!=null){const tr=document.createElement('tr');tr.append(text('td','MSE'),text('td','—','missing'),text('td',number(row.entire_mse)));$('metric-rows').append(tr);}
  $('relations').replaceChildren();
  for(const [label,names] of [['上游',row.parents],['下游',row.children]]){
    const head=text('div','','relation-head');head.append(text('span',label),text('span',String(names.length)));$('relations').append(head);
    for(const name of names){const target=byName.get(name),display=layerName(target)||name,button=text('button',short(display),'relation');button.title=display+'\n'+(target?.tensor_name||name);button.onclick=()=>choose(target,true);$('relations').append(button);}
    if(!names.length)$('relations').append(text('div','—','legend-text'));
  }
  svg.querySelector('#selected-outline')?.remove();
  const node=svg.querySelector(`image.node[data-i="${rows.indexOf(row)}"]`);
  if(node){
    const box=node.getBBox(),outline=document.createElementNS(ns,'rect');
    Object.entries({id:'selected-outline',x:box.x-4,y:box.y-4,width:box.width+8,height:box.height+8,rx:15,fill:'none',stroke:'#0c6f70','stroke-width':2,'pointer-events':'none'}).forEach(([key,value])=>outline.setAttribute(key,value));svg.append(outline);
    if(focus){const current=view();setView([box.x+box.width/2-current[2]/2,box.y+box.height/2-current[3]/2,current[2],current[3]]);}
  }
}
function setPanel(name,open){
  $('app').classList.toggle(`hide-${name}`,!open);$(`toggle-${name}`).setAttribute('aria-expanded',String(open));
  $(`${name}-label`).textContent=(open?'收起':'展开')+(name==='layers'?'列表':'详情');
}
['layers','details'].forEach(name=>$(`toggle-${name}`).onclick=()=>setPanel(name,$('app').classList.contains(`hide-${name}`)));
setPanel('layers',window.innerWidth>600);setPanel('details',window.innerWidth>900);
$('model-title').textContent=payload.title;$('model-title').title=payload.title;
$('graph-count').textContent=`${rows.length} 个节点 · ${payload.edges} 条连接`;
const covered=layers.filter(row=>row.single_cos!=null).length;
$('coverage').textContent=`单层指标覆盖 ${covered} / ${layers.length} 层`;$('coverage-bar').style.width=`${covered/Math.max(layers.length,1)*100}%`;
const [eucMin,eucMax]=payload.euc_range,eucSpan=eucMax-eucMin;
const hasEuc=rows.some(row=>Number.isFinite(row.entire_euc));
if(!hasEuc||eucSpan===0){
  $('euc-legend').classList.add('single-value');$('euc-legend').querySelector('.legend-bar').style.background=hasEuc?'#3cb371':'#b0b0b0';
  $('euc-ticks').append(text('span',hasEuc?compact(eucMin):'未提供'));
}else for(const fraction of [0,.25,.5,1]){
  const tick=text('span',compact(eucMin+fraction*eucSpan));tick.style.left=`${fraction*100}%`;tick.title=number(eucMin+fraction*eucSpan);$('euc-ticks').append(tick);
}
const [minX,maxX,minY,maxY]=payload.bounds,figureH=maxY-minY||1,ratio=svg.clientWidth/Math.max(svg.clientHeight,1);
// 小图适配全图；长图保留局部初始视图。
const initialH=rows.length<=12?Math.max(figureH+70,(maxX-minX+70)/ratio):figureH/10;
baseView=[(minX+maxX-initialH*ratio)/2,rows.length<=12?(minY+maxY-initialH)/2:minY-initialH*.1,initialH*ratio,initialH];
svg.setAttribute('viewBox',baseView.join(' '));
const mini=$('map-svg'),nodes=[...svg.querySelectorAll('image.node')];
let bounds=[minX-30,minY-30,maxX-minX+60,maxY-minY+60];
if(rows.length<=12){const left=Math.min(bounds[0],baseView[0])-20,top=Math.min(bounds[1],baseView[1])-20;bounds=[left,top,Math.max(bounds[0]+bounds[2],baseView[0]+baseView[2])+20-left,Math.max(bounds[1]+bounds[3],baseView[1]+baseView[3])+20-top];}
mini.setAttribute('viewBox',bounds.join(' '));
for(const node of nodes){const box=document.createElementNS(ns,'rect');for(const key of ['x','y','width','height'])box.setAttribute(key,node.getAttribute(key));box.setAttribute('rx',10);box.setAttribute('fill','#bdccd5');mini.append(box);}
const viewport=document.createElementNS(ns,'rect');viewport.classList.add('viewport');mini.append(viewport);syncView();
let mapDrag;
const map=$('map');
function moveMap(event){const p=point(mini,event);setView([p.x-mapDrag.x,p.y-mapDrag.y,mapDrag.w,mapDrag.h]);}
map.onpointerdown=event=>{
  if(event.button!==0||!event.isPrimary||mapDrag)return;
  const p=point(mini,event),current=view(),inside=p.x>=current[0]&&p.x<=current[0]+current[2]&&p.y>=current[1]&&p.y<=current[1]+current[3];
  mapDrag={id:event.pointerId,x:inside?p.x-current[0]:current[2]/2,y:inside?p.y-current[1]:current[3]/2,w:current[2],h:current[3]};
  map.setPointerCapture(event.pointerId);map.classList.add('dragging');if(!inside)moveMap(event);event.preventDefault();
};
map.onpointermove=event=>{if(mapDrag?.id===event.pointerId){moveMap(event);event.preventDefault();}};
function endMap(event){if(mapDrag?.id!==event.pointerId)return;mapDrag=null;map.classList.remove('dragging');if(map.hasPointerCapture(event.pointerId))map.releasePointerCapture(event.pointerId);}
['pointerup','pointercancel','lostpointercapture'].forEach(type=>map.addEventListener(type,endMap));
svg.addEventListener('wheel',event=>{event.preventDefault();const p=point(svg,event);zoomAt(p.x,p.y,event.deltaY>0?1.15:1/1.15);},{passive:false});
const pointers=new Map();let pan,pinch,moved=false,pressedNode;
const distance=()=>{const [a,b]=[...pointers.values()];return Math.hypot(a.x-b.x,a.y-b.y);};
function startPan(){const p=[...pointers.values()][0];pan={x:p.x,y:p.y,view:view(),inverse:svg.getScreenCTM().inverse()};}
svg.onpointerdown=event=>{
  if(event.pointerType==='mouse'&&event.button!==0)return;
  svg.focus({preventScroll:true});
  if(!pointers.size)pressedNode=event.target.getAttribute('data-i');
  pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});svg.setPointerCapture(event.pointerId);
  if(pointers.size===1){moved=false;startPan();svg.style.cursor='grabbing';}
  else if(pointers.size===2){pinch=distance();pan=null;moved=true;}event.preventDefault();
};
svg.onpointermove=event=>{
  if(!pointers.has(event.pointerId))return;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
  if(pointers.size===2){const d=distance(),[a,b]=[...pointers.values()],p=point(svg,{clientX:(a.x+b.x)/2,clientY:(a.y+b.y)/2});if(d>0&&pinch>0)zoomAt(p.x,p.y,pinch/d);pinch=d;return;}
  if(pan){const delta=new DOMPoint(event.clientX-pan.x,event.clientY-pan.y).matrixTransform(new DOMMatrix([pan.inverse.a,pan.inverse.b,pan.inverse.c,pan.inverse.d,0,0]));if(Math.hypot(event.clientX-pan.x,event.clientY-pan.y)>4)moved=true;setView([pan.view[0]-delta.x,pan.view[1]-delta.y,pan.view[2],pan.view[3]]);}
};
function endPointer(event){
  const clicked=event.type==='pointerup'&&!moved&&pointers.size===1?pressedNode:null;
  if(!pointers.delete(event.pointerId))return;pinch=null;if(pointers.size===1)startPan();else{pan=null;svg.style.cursor='grab';}
  if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);
  if(clicked!=null)choose(rows[Number(clicked)]);
}
['pointerup','pointercancel','lostpointercapture'].forEach(type=>svg.addEventListener(type,endPointer));
const pressed=new Set(),editing=target=>target instanceof Element&&!!target.closest('input,select,textarea,[contenteditable="true"]');
window.addEventListener('keydown',event=>{if(editing(event.target)||event.ctrlKey||event.metaKey||event.altKey)return;const key=event.key.toLowerCase();if('wasd'.includes(key)&&key.length===1){pressed.add(key);event.preventDefault();}});
window.addEventListener('keyup',event=>pressed.delete(event.key.toLowerCase()));
window.addEventListener('blur',()=>pressed.clear());document.addEventListener('focusin',event=>{if(editing(event.target))pressed.clear();});
function animate(){if(pressed.size){const current=view(),dx=(pressed.has('d')-pressed.has('a'))*payload.pan_speed,dy=(pressed.has('s')-pressed.has('w'))*payload.pan_speed;setView([current[0]+dx,current[1]+dy,current[2],current[3]]);}requestAnimationFrame(animate);}animate();
$('search').oninput=renderList;$('sort').onchange=renderList;$('fit').onclick=()=>setView(baseView.slice());
for(const [id,factor] of [['zoom-in',1/1.15],['zoom-out',1.15]])$(id).onclick=()=>{const current=view();zoomAt(current[0]+current[2]/2,current[1]+current[3]/2,factor);};
$('copy-name').disabled=!selected;
$('copy-name').onclick=async()=>{
  if(!selected)return;
  try{await navigator.clipboard.writeText(layerName(selected));$('toast').textContent='完整层名已复制';}
  catch{const range=document.createRange();range.selectNodeContents($('detail-name'));window.getSelection().removeAllRanges();window.getSelection().addRange(range);$('toast').textContent='已选中层名，可按 Ctrl+C 复制';}
  $('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',1800);
};
renderList();choose(selected);
if(!rows.length){$('canvas-empty').hidden=false;$('detail-name').textContent='未选择层';}

</script>
</body>
</html>'''

    def render(self, show: bool = True) -> str:
        """生成单文件交互报告，节点布局及连接路径保持原有算法。"""
        layout = self._layout_engine()
        real_pos, dummy_pos, edge_routes = {}, {}, {}
        if self.layer_row:
            depth = layout._layers()
            if depth:
                real_pos, dummy_pos, edge_routes, _ = layout._compute_layout(depth)
        paths = self._build_edge_paths(real_pos, dummy_pos, edge_routes, layout=layout)
        node_w, node_h = self.NODE_WIDTH, self.NODE_HEIGHT
        all_xs = [p[0] for p in real_pos.values()] + [p[0] for p in dummy_pos.values()]
        all_ys = [p[1] for p in real_pos.values()] + [p[1] for p in dummy_pos.values()]
        bounds = [min(all_xs, default=0)-node_w/2, max(all_xs, default=0)+node_w/2,
                  min(all_ys, default=0)-node_h/2, max(all_ys, default=0)+node_h/2]
        edges_svg = ''.join(
            f'<path d="{self._netron_curve_path(points)}" fill="none" stroke="{color}" '
            'stroke-width="1.6" marker-end="url(#arrow)"/>' for points, color, _ in paths)
        rows, nodes_svg = [], []
        for name in sorted(self.layer_row, key=self.node_order.__getitem__):
            row = self.layer_row[name]
            image, _ = self._svg_node(name, row.get('op_type', '?'), row)
            x, y = real_pos[name]
            nodes_svg.append(
                f'<image href="{image}" x="{x-node_w/2:.1f}" y="{y-node_h/2:.1f}" '
                f'width="{node_w:g}" height="{node_h:g}" data-i="{len(rows)}" class="node"/>')
            rows.append(dict(row, parents=self.parents.get(name, []), children=self.children.get(name, [])))
        svg = (
            '<svg id="svg" tabindex="0" aria-label="精度计算图" preserveAspectRatio="xMidYMid meet">\n'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#666"/></marker></defs>\n'
            + edges_svg + ''.join(nodes_svg) + '</svg>')
        payload = dict(rows=rows, title=self.title, edges=len(paths), bounds=bounds,
                       euc_range=[self.euc_color_min, self.euc_color_max], pan_speed=self.pan_speed)
        encoded = json.dumps(payload, ensure_ascii=False, default=float, allow_nan=False).replace('<', '\\u003c')
        parts = {'__TITLE__': html.escape(self.title), '__PAYLOAD__': encoded, '__SVG__': svg}
        document = re.sub(r'__TITLE__|__PAYLOAD__|__SVG__', lambda match: parts[match[0]], self._REPORT_HTML)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(document, encoding='utf-8')
        print(f"Network analysis HTML saved to: {self.output_path}")
        if show:
            import webbrowser
            webbrowser.open(self.output_path.resolve().as_uri())
        return str(self.output_path)



# ----------------------------------------------------------------------
# CLI：需 SDK 的读取子命令（供手工核对，也可被外部以子进程调用）
#   python accuracy_debugger.py encodings <dlc> <out_json>
#   python accuracy_debugger.py layers    <dlc> <out_json>
#   python accuracy_debugger.py tinydlc   <dlc> <out_json>
#   python accuracy_debugger.py tinydlc-batch <dlcs_list_json>
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
            QnnSdkTools.write_tiny_meta_batch(argv[1])
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
            QnnSdkTools.write_tiny_meta(dlc, out_json=out_json)
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
