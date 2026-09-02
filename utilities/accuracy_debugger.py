import re
import os
import sys
import shutil

from pathlib import Path
from typing import Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
import cv2


current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))
from utils import clean_files_or_dirs, run_command, MultThreadExetutor




def plot_accuracy_summary(names:list[str]=None, single_euc:np.ndarray=None, single_cos:np.ndarray=None, 
                            entire_euc:np.ndarray=None, entire_cos:np.ndarray=None, mse_vals=None, 
                            entire_val_color:str="blue", save_path=None):
    """统一的逐层精度可视化：欧氏距离柱状图 + 余弦相似度折线图（两个子图始终保留）。

    使用分支按传入数据绘制：只画实际存在的数据（None 则跳过该部分），
    即使某子图没有任何数据也会保留该子图（显示标题、坐标轴与网格，不报错）。

    Args:
        names: x 轴刻度标签（可为 None）。
        single_euc: 欧氏距离柱状数据（逐层单层误差，可为 None）。
        single_cos: 余弦相似度折线数据（逐层单层余弦，可为 None）。
        entire_euc: 可选累积欧氏距离（entire，自输入累计），以折线叠加在上图。
        entire_cos: 可选累积余弦相似度（entire），以折线叠加在下图。
        mse_vals: 可选 MSE 右轴折线数据。
        save_path: 保存路径。
    """
    from matplotlib import axes
    import matplotlib.pyplot as plt

    # 数据长度：从任一可用数组推断（names 优先），用于 x 轴范围与刻度
    n = 0
    for arr in (names, single_euc, single_cos, entire_euc, entire_cos, mse_vals):
        if arr is not None:
            n = len(arr)
            break
    layer_index = np.arange(n)

    # 两个子图始终创建（即使无数据也保留）
    fig, (ax_euc, ax_cos) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax_euc:axes.Axes
    ax_cos:axes.Axes

    # ---- 上图：欧氏距离（分支：只画有数据的部分）----
    extra_axes: list[axes.Axes] = []
    has_euc_data = False

    if single_euc is not None:
        single_euc = np.asarray(single_euc, dtype=np.float64)
        # 数据清洗：避免 log(0) 报错，将绝对的 0 值替换为极小值
        euc_plot = single_euc.copy()
        euc_plot[euc_plot == 0] = 1e-10
        ax_euc.set_yscale('linear')
        ax_euc.bar(layer_index, euc_plot, color='skyblue', edgecolor='black',
                    linewidth=0.5, alpha=0.7, label='Euc Dist Per Layer')
        has_euc_data = True

    if entire_euc is not None:
        # 累积欧氏距离与单层欧氏距离使用同一坐标刻度
        entire_euc_plot = np.asarray(entire_euc, dtype=np.float64).copy()
        entire_euc_plot[entire_euc_plot == 0] = 1e-10
        ax_euc.plot(layer_index, entire_euc_plot, color=entire_val_color, marker='.',
                    linestyle='-', linewidth=1.5, markersize=2, label='Euc (entire)')
        has_euc_data = True

    if mse_vals is not None:
        # 右侧纵轴：MSE（量级可能千分之一~个位，与欧氏距离的几十上百分离显示）
        ax_mse = ax_euc.twinx()
        ax_mse.plot(layer_index, mse_vals, color='purple', marker='.', linestyle='-',
                    linewidth=1, markersize=2, label='MSE Per Layer')
        ax_mse.set_ylabel('MSE', fontsize=12)
        extra_axes.append(ax_mse)
        has_euc_data = True

    ax_euc.set_title(f'Euclidean Distance', fontsize=14, fontweight='bold')
    ax_euc.set_ylabel('Euclidean Distance', fontsize=12)
    ax_euc.grid(True, which="both", ls="--", alpha=0.5)
    if has_euc_data:
        # 合并右轴（MSE）与左轴（柱状 / entire）的图例
        lines, labels = ax_euc.get_legend_handles_labels()
        for ax in extra_axes:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab
        if lines:
            ax_euc.legend(lines, labels, loc='upper left')

    # ---- 下图：余弦相似度（分支：只画有数据的部分）----
    has_cos_data = False

    if single_cos is not None:
        single_cos = np.asarray(single_cos, dtype=np.float64)
        # 数据清洗：将恰好等于 0 的余弦值置为 NaN，matplotlib 会跳过这些点且不连线，
        # 避免异常的 0 值把 y 轴显示范围压扁（正常余弦值集中在 0.98~1.0 附近）
        cos_sim_plot = np.where(single_cos == 0.0, np.nan, single_cos)
        single_label = 'Cosine (single)' if entire_cos is not None else 'Per Layer'
        ax_cos.set_yscale('linear')
        ax_cos.plot(layer_index, cos_sim_plot, color='green', marker='.',
                    linestyle='-', linewidth=1, markersize=2, label=single_label)
        has_cos_data = True

    if entire_cos is not None:
        entire_cos_plot = np.where(np.asarray(entire_cos) == 0.0, np.nan, np.asarray(entire_cos))
        ax_cos.plot(layer_index, entire_cos_plot, color=entire_val_color, marker='.',
                    linestyle='-', linewidth=1.5, markersize=2, label='Cosine (entire)')
        has_cos_data = True

    ax_cos.set_title('Cosine Similarity', fontsize=14, fontweight='bold')
    ax_cos.set_ylabel('Cosine Similarity', fontsize=12)
    ax_cos.set_xticks(layer_index)
    if names is not None:
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

    def read_error_analysis(self) -> list[dict]:
        """
        读取 RKNN 精度分析结果文件 (snapshot/error_analysis.txt)。

        Args:
            error_analysis_path (str | None): 精度分析结果文件路径。
                - None (默认): 使用 self.tmp_dir/snapshot/error_analysis.txt。

        Returns:
            list[dict]: 逐层精度数据列表，每个元素包含:
                - op_type (str): 算子类型，如 'Conv'、'LeakyRelu'
                - layer_name (str): 层名称
                - entire_cos (float | None): 累积余弦相似度（从输入累计到该层）
                - entire_euc (float | None): 累积欧氏距离
                - single_cos (float | None): 单层余弦相似度（反映该层自身量化误差）
                - single_euc (float | None): 单层欧氏距离
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

        rows: list[dict] = []
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

                rows.append({
                    'op_type': op_type,
                    'layer_name': layer_name,
                    'entire_cos': entire_cos,
                    'entire_euc': entire_euc,
                    'single_cos': single_cos,
                    'single_euc': single_euc,
                })

        print(f"Loaded {len(rows)} layers from {error_analysis_path}")
        return rows

    def plot_accuracy_analysis(self):
        """
        读取并可视化 RKNN 精度分析结果（欧氏距离柱状图 + 余弦相似度折线图）。
        """
        from matplotlib import axes
        import matplotlib.pyplot as plt

        rows = self.read_error_analysis()
        if not rows:
            print("No data to plot.")
            return rows

        layer_names = [r['layer_name'] for r in rows]
        n = len(rows)
        x = np.arange(n)

        entire_cos = np.array([r['entire_cos'] for r in rows], dtype=float)
        entire_euc = np.array([r['entire_euc'] for r in rows], dtype=float)
        single_cos = np.array([r['single_cos'] for r in rows], dtype=float)
        single_euc = np.array([r['single_euc'] for r in rows], dtype=float)

        save_path = self.tmp_work_dir / 'rknn_accuracy_analysis_summary.png'

        plot_accuracy_summary(
            names=layer_names,
            single_euc=single_euc,
            single_cos=single_cos,
            entire_euc=entire_euc,
            entire_cos=entire_cos,
            entire_val_color="orange",
            save_path=save_path,
        )

        self.file_or_dir_to_clean.append(save_path)


    # ------------------------------------------------------------------
    # 带路径追踪的精度分析：从多个输入到多个输出的排列组合路径
    # ------------------------------------------------------------------

    def _build_onnx_tensor_graph(self) -> dict:
        """
        从 ONNX 模型构建张量级 DAG（用于路径追踪）。

        使用 self.tmp_model_path（convert() 复制到 tmp 目录的模型副本）。

        Returns:
            dict: 包含
                - tensor_set (set[str]): 全部张量名（输入/输出/节点输出）
                - pred (dict[str, list[str]]): 张量 -> 产生它的节点输入张量列表
                - succ (dict[str, list[str]]): 张量 -> 消费它的后续张量列表
                - inputs (list[str]): 模型输入名
                - outputs (list[str]): 模型输出名
        """
        import onnx

        model = onnx.load(str(self.tmp_model_path))
        g = model.graph

        tensor_set: set[str] = {t.name for t in list(g.input) + list(g.output)}
        pred: dict[str, list[str]] = {}
        succ: dict[str, list[str]] = {}
        for node in g.node:
            outs = [o for o in node.output if o]
            ins = [i for i in node.input if i]
            for out in outs:
                tensor_set.add(out)
                pred.setdefault(out, []).extend(ins)
            for i in ins:
                succ.setdefault(i, []).extend(outs)

        return {
            'tensor_set': tensor_set,
            'pred': pred,
            'succ': succ,
            'inputs': [t.name for t in g.input],
            'outputs': [t.name for t in g.output],
        }

    def _match_tensor(self, layer_name: str, tensor_set: set[str]) -> str | None:
        """
        将 RKNN 快照层名匹配到 ONNX 张量名。

        RKNN 会在原始张量名后追加后缀（如 _sw、-rs、_mm 等），
        因此先尝试精确匹配，失败则取"最长的、是该层名前缀的 ONNX 张量名"。

        Returns:
            str | None: 匹配到的 ONNX 张量名，未匹配返回 None。
        """
        if layer_name in tensor_set:
            return layer_name

        best: str | None = None
        for t in tensor_set:
            if layer_name.startswith(t):
                if best is None or len(t) > len(best):
                    best = t
        return best

    def _build_layer_graph(self, rows: list[dict], graph: dict) -> tuple[dict, dict, dict]:
        """
        构建"层图"：以 RKNN 快照层为节点，边由 ONNX 张量依赖关系推导。

        - 同一 ONNX 张量对应的多个层（如 template / template_int8 / template_conv，
          RKNN 的输入/输出处理链）按快照顺序串联。
        - 沿 ONNX succ 广度优先，跳过没有快照层的中间张量，连到下一个有快照层的张量。

        Returns:
            tuple:
                - children (dict[str, list[str]]): 层 -> 下游层列表
                - parents (dict[str, list[str]]): 层 -> 上游层列表
                - tensor_layers (dict[str, list[str]]): ONNX 张量 -> 对应层列表
        """

        tensor_set = graph['tensor_set']
        succ = graph['succ']

        tensor_layers: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            r['tensor'] = self._match_tensor(r['layer_name'], tensor_set)
            tensor_layers[r['tensor']].append(r['layer_name'])

        node_order = {r['layer_name']: i for i, r in enumerate(rows)}
        children: dict[str, list[str]] = defaultdict(list)

        def add_edge(a: str | None, b: str | None) -> None:
            if a and b and a != b:
                children[a].append(b)

        # 同一张量的层链（RKNN 输入/输出处理层）。
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

    def _build_terminal_nodes(self, rows: list[dict], children: dict, parents: dict, graph: dict):
        """
        在真实层之外追加 Output 示意终端节点并连边（模仿 SnpeAccuracyDebugger）。

        RKNN 精度分析结果默认已包含输入层，因此这里只追加输出节点：
        每个模型输出追加一个无精度数据的示意 'Output' 终端节点
        （不替换/不改写原输出层），从真正的输出层连出；若该输出张量未被快照，
        则反向 BFS 找到能到达它的最深快照层连出。命名加 '(out)' 后缀避免与原层重名。

        Returns:
            tuple:
                - output_rows (list[dict]): Output 示意节点行
                - output_layers (list[str]): Output 节点名
                - aug_children (dict[str, list[str]]): 加入终端节点边后的 children
                - aug_parents (dict[str, list[str]]): 加入终端节点边后的 parents
        """

        tensor_layers = defaultdict(list)
        for r in rows:
            t = r.get('tensor')
            if t is not None:
                tensor_layers[t].append(r['layer_name'])
        # 同一张量可能有多个变体层（如 output0-rs_tp / output0-rs / output0_int8 /
        # output0），处理链的链尾(lst[-1])才是真正的最终输出层，输出终端应从它连出
        tensor_to_layer = {t: lst[-1] for t, lst in tensor_layers.items()}
        node_order = {r['layer_name']: i for i, r in enumerate(rows)}

        aug_children = defaultdict(list)
        aug_parents = defaultdict(list)
        for a, cl in children.items():
            aug_children[a] = list(cl)
            for b in cl:
                aug_parents[b].append(a)

        pred = graph['pred']

        existing_names = set(node_order)
        output_layers: list[str] = []
        output_rows: list[dict] = []
        for out in graph['outputs']:
            # 原输出层通常已占用输出张量名，示意节点加后缀避免重名
            node_name = out if out not in existing_names else f'{out} (out)'
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
            output_rows.append({
                'layer_name': node_name,
                'op_type': 'Output',
                'entire_cos': None,
                'entire_euc': None,
                'single_cos': None,
                'single_euc': None,
            })

        return output_rows, output_layers, aug_children, aug_parents

    def read_path_analysis(self) -> dict:
        """
        读取 RKNN 精度分析数据，并结合 ONNX 图结构追踪"输入 -> 输出"的排列组合路径。

        Returns:
            dict: 包含
                - inputs (list[str]): 模型输入名
                - outputs (list[str]): 模型输出名
                - paths (dict[tuple[str, str], list[dict]]): (输入, 输出) -> 路径层列表，
                  每层包含 op_type / layer_name / tensor / entire_cos / entire_euc /
                  single_cos / single_euc（累积精度与单层精度）。
        """

        rows = self.read_error_analysis()
        if not rows:
            return {}

        graph = self._build_onnx_tensor_graph()
        children, parents, _ = self._build_layer_graph(rows, graph)

        node_order = {r['layer_name']: i for i, r in enumerate(rows)}
        layer_row = {r['layer_name']: r for r in rows}

        def descendants(start: str) -> set[str]:
            """start 出发能到达的所有层（不含自身）。"""
            seen: set[str] = set()
            dq = deque([start])
            while dq:
                n = dq.popleft()
                for c in children.get(n, []):
                    if c not in seen:
                        seen.add(c)
                        dq.append(c)
            return seen

        def ancestors(start: str) -> set[str]:
            """能到达 start 的所有层（不含自身）。"""
            seen: set[str] = set()
            dq = deque([start])
            while dq:
                n = dq.popleft()
                for p in parents.get(n, []):
                    if p not in seen:
                        seen.add(p)
                        dq.append(p)
            return seen

        # 输入/输出层：ONNX 图输入输出名对应的快照层
        input_layers = [t for t in graph['inputs'] if t in layer_row]
        output_layers = [t for t in graph['outputs'] if t in layer_row]

        paths: dict[tuple[str, str], list[dict]] = {}
        for inp in input_layers:
            for out in output_layers:
                if inp == out:
                    continue
                # 路径层 = (从输入可到达) ∩ (可到达输出)，并补上输入/输出层自身
                path_layers = sorted(
                    (descendants(inp) | {inp}) & (ancestors(out) | {out}),
                    key=lambda x: node_order[x],
                )
                paths[(inp, out)] = [
                    {
                        'op_type': layer_row[ln]['op_type'],
                        'layer_name': ln,
                        'tensor': layer_row[ln].get('tensor'),
                        'entire_cos': layer_row[ln]['entire_cos'],
                        'entire_euc': layer_row[ln]['entire_euc'],
                        'single_cos': layer_row[ln]['single_cos'],
                        'single_euc': layer_row[ln]['single_euc'],
                    }
                    for ln in path_layers
                ]

        result = {
            'inputs': input_layers,
            'outputs': output_layers,
            'paths': paths,
            'rows': rows,
        }
        return result

    def draw_network_analysis(self, show:bool=True):
        """
        带路径追踪的精度分析（Netron 风格网络图）。

        复用 read_path_analysis() 的数据，并重新构建层图（children/parents），
        交给独立的 pyvis 可视化类 AccuracyGraph 渲染整个网络图。

        节点填充色 = 累积精度 entire_cos（红-黄-绿），悬停查看单层精度与欧氏距离。
        额外追加 Output 示意终端节点（RKNN 默认已含输入层，无需额外加输入）。

        Returns:
            dict: read_path_analysis() 的原始结果。
        """
        data = self.read_path_analysis()
        if not data or not data['paths']:
            print("No path data to plot.")
            return data

        graph = self._build_onnx_tensor_graph()
        children, parents, _ = self._build_layer_graph(data['rows'], graph)

        # 追加 Output 示意终端节点
        output_rows, output_layers, children, parents = self._build_terminal_nodes(
            data['rows'], children, parents, graph)
        data['rows'] = data['rows'] + output_rows
        data['outputs'] = output_layers

        output_path = self.tmp_work_dir / 'rknn_graph_accuracy_analysis.html'
        # self.file_or_dir_to_clean.append(output_path)

        viz = AccuracyGraph(
            data=data,
            children=children,
            parents=parents,
            output_path=output_path,
            title='RKNN Graph Accuracy Analysis',
        )

        html_path = viz.render(show=show)
        self.file_or_dir_to_clean.append(html_path)
        return

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

    def __init__(self, tmp_dir:str, onnx_path:str=None, debugger_picture_list:list=None, qairt_script=None):

        from onnx_to_qnn import QAIRTScript
        
        self.tmp_dir = Path(tmp_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = self.tmp_dir / 'qnn_accuracy_analysis'
        self.onnx_path = Path(onnx_path).resolve()
        self.debugger_picture_list = [Path(p).resolve() for p in (debugger_picture_list or []) if Path(p).exists()]
        
        self.qairt_script:QAIRTScript = qairt_script
        self.qairt_sdk_root = Path(getattr(self.qairt_script, 'qairt_sdk_root', None) or os.environ.get('QAIRT_SDK_ROOT'))

        if self.qairt_script is not None:
            self.exe_qnn_net_run = self.qairt_script.get_tool('qnn-net-run')
            self.platform_arch = self.qairt_script.current_platform_arch()
            self.backend_lib_golden = self.qairt_script.get_lib('libQnnCpu')
            # HTP 后端在 x86_64 Windows 上无执行路径（无 QEMU 模拟器、无平台 stub/skel，
            # 仅 aarch64/arm64x Windows 有真实 NPU），QnnHtp.dll 初始化会让 qnn-net-run 卡死。
            # x86_64 Linux 通过 QEMU 模拟器可运行，故仅 x86_64 Windows 回退到 CPU 后端。
            if self.platform_arch == 'x86_64-windows-msvc':
                print("[QAIRTAccuracyDebugger] x86_64-windows-msvc: HTP backend unavailable "
                      "(no simulator/platform libs); using CPU backend for target DLC too.")
                self.backend_lib_target = self.backend_lib_golden
            else:
                self.backend_lib_target = self.qairt_script.get_lib('libQnnHtp')
        else:
            self.platform_arch = 'x86_64-linux-clang'
            self.exe_qnn_net_run = str(self.qairt_sdk_root / 'bin' / self.platform_arch / 'qnn-net-run')
            self.backend_lib_golden = self.qairt_sdk_root / 'lib' / self.platform_arch / 'libQnnCpu.so'
            self.backend_lib_target = self.qairt_sdk_root / 'lib' / self.platform_arch / 'libQnnHtp.so'

        self.onnx_info: dict = {}
        self.file_or_dir_to_clean: list[str] = [str(self.working_dir)]

    # ------------------------------------------------------------------
    # 模型信息 / 通用工具
    # ------------------------------------------------------------------

    def set_model_info(self, onnx_info: dict):
        """设置模型输入/输出信息（prepare_input_data 需要 inputs 的名称与形状）。"""
        self.onnx_info = onnx_info or {}


    def prepare_input_data(self, mean_rgb: list = [[0, 0, 0]], std_rgb: list = [[1, 1, 1]],
                           set_input_order: str = 'nchw') -> str:
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
                if set_input_order == 'nchw':
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

    def run_qnn_net_run(self, dlc_path:str, backend_lib:str, input_list:str, output_dir: str) -> int:
        """执行 qnn-net-run --dlc_path <dlc> --backend <lib> --input_list --output_dir [--debug]。"""
        qnn_net_run = self.exe_qnn_net_run

        cmd = f'{qnn_net_run} --backend "{backend_lib}" --debug --log_level warn'
        cmd += f' --dlc_path {dlc_path} --input_list {input_list} --output_dir {output_dir}'

        return run_command(cmd, signature="[QAIRTAccuracyDebugger]")

    # ------------------------------------------------------------------
    # 主入口：双 DLC 对比
    # ------------------------------------------------------------------

    def accuracy_analysis(self, golden_dlc_path: str, target_dlc_path: str,
                          mean_rgb: list = [[0, 0, 0]], std_rgb: list = [[1, 1, 1]],
                          set_input_order: str = 'nchw') -> int:
        """双 DLC 精度对比主入口：
        1) qnn-net-run 跑 FP32 DLC（CPU 后端）拿每层输出；
        2) qnn-net-run 跑量化 DLC（HTP 后端）拿每层输出；
        3) 逐层计算 cosine / 欧氏距离 / mse；
        4) 渲染 PNG + (可选 HTML) + CSV。

        Args:
            golden_dlc_path: FP32（未量化）DLC 路径。
            target_dlc_path: 量化 DLC 路径。
            backend_golden / backend_target: 覆盖默认后端 lib 名。
        """
        self.working_dir.mkdir(parents=True, exist_ok=True)

        input_list = self.prepare_input_data(mean_rgb, std_rgb, set_input_order)

        names, entire_cos, entire_euc, mse_vals = self.entire_accuracy_analysis(golden_dlc_path, target_dlc_path, input_list)

        # 统计显示 1：PNG 双子图（欧氏距离柱状 + 余弦折线）
        plt_path = self.tmp_dir / 'qnn_accuracy_analysis_summary.png'
        plot_accuracy_summary(
            names=names,
            entire_euc=entire_euc,
            entire_cos=entire_cos,
            mse_vals=mse_vals,
            entire_val_color="blue",
            save_path=plt_path,
        )
        self.file_or_dir_to_clean.append(plt_path)

        # 统计显示 2：AccuracyGraph HTML（若有 ONNX 可建图结构）
        if self.onnx_path is not None:
            self._plot_network_analysis_entire_only(
                names=names, entire_cos=entire_cos, entire_euc=entire_euc, mse_vals=mse_vals,
            )

        return 0

    def clean(self):
        clean_files_or_dirs(self.file_or_dir_to_clean)


    def entire_accuracy_analysis(self, golden_dlc_path:str, target_dlc_path:str, input_list:str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
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
        # ret = self.run_qnn_net_run(golden_dlc_path, self.backend_lib_golden, input_list, str(golden_dir))
        # if ret != 0:
        #     raise RuntimeError(f"qnn-net-run golden DLC failed with code {ret}")
        
        # ret = self.run_qnn_net_run(target_dlc_path, self.backend_lib_target, input_list, str(target_dir))
        # if ret != 0:
        #     raise RuntimeError(f"qnn-net-run target DLC failed with code {ret}")

        names, entire_cos, entire_euc, mse_vals = self._compare_dlc_outputs(golden_dir, target_dir)
        return names, entire_cos, entire_euc, mse_vals

    def _compare_dlc_outputs(self, golden_dir: Path, target_dir: Path):
        """读两个 Result_0 目录，按张量名（sanitized 文件名）匹配，算 cos/euc/mse。"""
        def load(d):
            out = {}
            for p in Path(d).rglob('*.raw'):
                out[p.stem] = np.fromfile(p, dtype=np.float32)
            return out

        gold = load(golden_dir)
        targ = load(target_dir)
        common = {n for n in gold if n in targ and gold[n].size == targ[n].size}
        # 优先按 ONNX 计算图顺序排序；不在图中的额外张量（如 hybrid 转换节点）排后面
        if self.onnx_path is not None:
            graph = self._build_onnx_tensor_graph()
            order = [n for n in graph['order'] if n in common]
            order += sorted(common - set(order))
        else:
            order = sorted(common)
        print(f"[QAIRTAccuracyDebugger] matched layers: {len(order)}")

        names, cos, euc, mse = [], [], [], []
        for n in order:
            a, b = gold[n], targ[n]
            names.append(n)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            euc.append(float(np.linalg.norm(a - b)))
            mse.append(float(np.mean((a - b) ** 2)))
        return names, np.asarray(cos), np.asarray(euc), np.asarray(mse)

    # ------------------------------------------------------------------
    # 渲染：AccuracyGraph（HTML）与结果组织
    # ------------------------------------------------------------------

    def _plot_network_analysis_entire_only(self, names, entire_cos, entire_euc, mse_vals=None):
        """仅基于 entire（累积）数据渲染 AccuracyGraph：single_cos/single_euc 置 None。"""
        graph = self._build_onnx_tensor_graph()
        node_info = graph['node_info']

        results = []
        for i, name in enumerate(names):
            raw_stem = str(name)
            if raw_stem.endswith('.raw'):
                raw_stem = raw_stem[:-4]
            tensor = self._match_tensor(raw_stem, graph)
            results.append({
                'golden': str(name), 'infer': None, 'tensor': tensor,
                'layer_name': graph['tensors'].get(tensor, str(name)) if tensor is not None else str(name),
                'op_type': node_info.get(tensor)[0] if tensor is not None and tensor in node_info else 'Unknown',
                'entire_cos': float(entire_cos[i]),
                'entire_euc': float(entire_euc[i]),
                'single_cos': None, 'single_euc': None,
                'mse': float(mse_vals[i]) if mse_vals is not None else None,
            })

        children, parents, _ = self._build_layer_graph(results, graph)
        self.draw_network_analysis(results, children, parents, graph, show=True)

    def draw_network_analysis(self, results: list, children: dict, parents: dict,
                              graph: dict, show: bool = True) -> dict:
        """接入 AccuracyGraph 渲染 Netron 风格网络图（HTML）。"""
        real_rows = [{
            'layer_name': r['layer_name'], 'op_type': r['op_type'],
            'entire_cos': r['entire_cos'], 'entire_euc': r['entire_euc'],
            'single_cos': r['single_cos'], 'single_euc': r['single_euc'],
        } for r in results]

        (input_rows, output_rows, input_layers, output_layers,
         aug_children, aug_parents) = self._build_terminal_nodes(results, children, parents, graph)

        data = {
            'rows': input_rows + real_rows + output_rows,
            'inputs': input_layers, 'outputs': output_layers, 'paths': {},
        }
        output_path = self.tmp_dir / 'qnn_graph_accuracy_analysis.html'
        viz = AccuracyGraph(data=data, children=aug_children, parents=aug_parents,
                            output_path=output_path, title='QNN Graph Accuracy Analysis')
        
        html_path = viz.render(show=show)
        self.file_or_dir_to_clean.append(html_path)
        return data

    def _build_onnx_tensor_graph(self) -> dict:
        """解析 self.onnx_path 的 ONNX 计算图，构建张量级 DAG（清洗名命名空间）。"""
        import onnx

        model = onnx.load(str(self.onnx_path))
        g = model.graph

        def sanitize(name: str) -> str:
            return re.sub(r'[/.]+', '_', name)

        tensors: dict[str, str] = {}
        for t in list(g.input) + list(g.output):
            tensors[sanitize(t.name)] = t.name
        for n in g.node:
            for o in n.output:
                if o:
                    tensors.setdefault(sanitize(o), o)

        node_info: dict[str, tuple[str, list[str]]] = {}
        pred: dict[str, list[str]] = {}
        succ: dict[str, list[str]] = {}
        for n in g.node:
            outs = [sanitize(o) for o in n.output if o]
            ins = [sanitize(i) for i in n.input if i]
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
                input_sizes[sanitize(t.name)] = s

        # ONNX 计算图遍历顺序（sanitized 张量名）：用于统计图/报告按图顺序排序
        order = []
        for n in g.node:
            for o in n.output:
                if o:
                    s = sanitize(o)
                    if s not in order:
                        order.append(s)
        for t in g.output:
            s = sanitize(t.name)
            if s not in order:
                order.append(s)

        return {
            'tensors': tensors,
            'node_info': node_info,
            'pred': pred,
            'succ': succ,
            'inputs': [sanitize(t.name) for t in g.input],
            'outputs': [sanitize(t.name) for t in g.output],
            'input_sizes': input_sizes,
            'order': order,
        }

    def _match_tensor(self, layer_name: str, graph: dict) -> str | None:
        """将 raw 文件名（不含 .raw）匹配到清洗后的 ONNX 张量名。"""
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

    def _build_layer_graph(self, rows: list[dict], graph: dict):
        """构建"层图"：以结果行为节点，边由 ONNX 张量依赖推导。"""
        succ = graph['succ']

        tensor_layers: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            tensor_layers[r['tensor']].append(r['layer_name'])

        node_order = {r['layer_name']: i for i, r in enumerate(rows)}
        children: dict[str, list[str]] = defaultdict(list)

        def add_edge(a: str | None, b: str | None) -> None:
            if a and b and a != b:
                children[a].append(b)

        for tin, lst in tensor_layers.items():
            if tin is None:
                continue
            for i in range(len(lst) - 1):
                add_edge(lst[i], lst[i + 1])

        for tin, lst in tensor_layers.items():
            if tin is None:
                continue
            l_in = lst[-1]
            visited: set[str] = set()
            dq = deque(succ.get(tin, []))
            while dq:
                t = dq.popleft()
                if t in visited:
                    continue
                visited.add(t)
                if t in tensor_layers:
                    add_edge(l_in, tensor_layers[t][0])
                    continue
                dq.extend(succ.get(t, []))

        for k in children:
            children[k] = sorted(set(children[k]), key=lambda x: node_order[x])

        parents: dict[str, list[str]] = defaultdict(list)
        for a, cl in children.items():
            for b in cl:
                parents[b].append(a)

        return children, parents, tensor_layers

    def _build_terminal_nodes(self, results: list[dict], children: dict, parents: dict, graph: dict):
        """在真实层之外追加 Input / Output 示意终端节点并连边。"""
        tensor_layers = defaultdict(list)
        for r in results:
            if r.get('tensor') is not None:
                tensor_layers[r['tensor']].append(r['layer_name'])
        tensor_to_layer = {t: lst[-1] for t, lst in tensor_layers.items()}
        node_order = {r['layer_name']: i for i, r in enumerate(results)}

        aug_children = defaultdict(list)
        aug_parents = defaultdict(list)
        for a, cl in children.items():
            aug_children[a] = list(cl)
            for b in cl:
                aug_parents[b].append(a)

        succ = graph['succ']
        pred = graph['pred']

        input_rows: list[dict] = []
        input_layers: list[str] = []
        for inp in graph['inputs']:
            onnx_name = graph['tensors'].get(inp, inp)
            input_layers.append(onnx_name)
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
                aug_children[onnx_name].append(tgt)
                aug_parents[tgt].append(onnx_name)
            input_rows.append({
                'layer_name': onnx_name, 'op_type': 'Input',
                'entire_cos': None, 'entire_euc': None,
                'single_cos': None, 'single_euc': None,
            })

        existing_names = set(node_order) | set(input_layers)
        output_layers: list[str] = []
        output_rows: list[dict] = []
        for out in graph['outputs']:
            onnx_out = graph['tensors'].get(out, out)
            node_name = onnx_out if onnx_out not in existing_names else f'{onnx_out} (out)'
            output_layers.append(node_name)
            if out in tensor_to_layer:
                producers = [tensor_to_layer[out]]
            else:
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
            output_rows.append({
                'layer_name': node_name, 'op_type': 'Output',
                'entire_cos': None, 'entire_euc': None,
                'single_cos': None, 'single_euc': None,
            })

        return input_rows, output_rows, input_layers, output_layers, aug_children, aug_parents



class AccuracyGraph:
    """
    基于 pyvis 的 RKNN 精度分析网络图（Netron 风格）。

    将 RKNN 快照层以"方框节点 + 箭头连线"呈现为从左到右分层的 DAG，
    节点填充色按累积精度 entire_cos 着色（红=差, 绿=良），悬停显示
    单层精度与欧氏距离等详细信息。

    Attributes:
        data (dict): read_path_analysis() 的返回结果。
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

    # RKNN 算子类型 -> Netron 类别
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
    }

    def __init__(
        self,
        data: dict,
        children: dict[str, list[str]],
        parents: dict[str, list[str]],
        output_path: str | Path,
        title: str = 'Accuracy Analysis',
    ):
        self.data = data
        self.children = children
        self.parents = parents
        self.output_path = Path(output_path)
        self.title = title
        self.node_sep = 50.0 # 横向：真实节点之间的间距
        self.edge_sep = 20.0 # 横向：长边虚拟节点占用的间距
        self.rank_sep = 50.0 # 纵向：层与层之间的间距
        self.pan_speed:float = 15

        self.rows = data.get('rows', [])
        self.inputs = data.get('inputs', [])
        self.outputs = data.get('outputs', [])
        self.paths = data.get('paths', {})
        self.node_order = {r['layer_name']: i for i, r in enumerate(self.rows)}
        self.layer_row = {r['layer_name']: r for r in self.rows}
        euc_values = [r.get('entire_euc') for r in self.rows
                  if r.get('entire_euc') is not None]
        self.euc_color_min = min(euc_values, default=0.0)
        self.euc_color_max = max(euc_values, default=1.0)

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
        """计算每层的拓扑深度（从各源点出发的最长路径），用作纵向分层坐标。

        采用 Kahn 拓扑排序，对边方向不依赖快照顺序（node_order），
        即便存在逆序边或环也能给出确定结果（环上节点兜底为深度 0）。
        """
        from collections import deque

        layer_row = self.layer_row
        depth: dict[str, int] = {}
        in_degree: dict[str, int] = {n: 0 for n in layer_row}
        for n in layer_row:
            for p in self.parents.get(n, []):
                if p in layer_row:
                    in_degree[n] += 1

        queue = deque(n for n in layer_row if in_degree[n] == 0)
        for n in queue:
            depth[n] = 0

        while queue:
            n = queue.popleft()
            for c in self.children.get(n, []):
                if c in layer_row and c not in depth:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        depth[c] = depth[n] + 1
                        queue.append(c)

        # 环上的节点不会被拓扑排序访问到，兜底置为 0
        for n in layer_row:
            if n not in depth:
                depth[n] = 0
        return depth

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

        real = set(self.layer_row.keys())

        # ---- 1. normalize：构建增强图（真实节点 + 虚拟节点） ----
        nodes: dict[str, dict] = {}
        for n in real:
            nodes[n] = {
                'rank': depth.get(n, 0),
                'width': node_w,
                'height': node_h,
                'dummy': False,
                'seq': self.node_order.get(n, 0),
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
            if a not in real:
                continue
            for b in blist:
                if b not in real:
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

        # ---- 2. order：交叉最小化 ----
        layering = self._dagre_order(nodes, aug_in, aug_out)

        # ---- 3. position：Brandes-Köpf ----
        x_pos, y_pos = self._dagre_position(
            layering, nodes, aug_in, aug_out, node_sep, edge_sep, rank_sep,
        )

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
        """重心法 + 交叉计数反馈 + transpose 交换，返回每层有序节点列表。"""
        from functools import cmp_to_key

        max_rank = max(nd['rank'] for nd in nodes.values())
        layering: list[list[str]] = [[] for _ in range(max_rank + 1)]
        for nid, nd in nodes.items():
            layering[nd['rank']].append(nid)
        for layer in layering:
            layer.sort(key=lambda nid: nodes[nid]['seq'])

        def cross_count(layers: list[list[str]]) -> int:
            total = 0
            for r in range(1, len(layers)):
                south_pos = {v: j for j, v in enumerate(layers[r])}
                entries = []
                for v in layers[r - 1]:
                    for w in aug_out.get(v, []):
                        if w in south_pos:
                            entries.append(south_pos[w])
                total += self._count_inversions(entries, len(layers[r]))
            return total

        def sort_layer(layer, neighbor_adj, neighbor_pos, bias_right):
            entries = []
            for idx, v in enumerate(layer):
                orders = [neighbor_pos[w] for w in neighbor_adj.get(v, []) if w in neighbor_pos]
                if orders:
                    entries.append({'v': v, 'bc': sum(orders) / len(orders), 'i': idx})
                else:
                    entries.append({'v': v, 'i': idx})
            sortable = [e for e in entries if 'bc' in e]
            unsortable = [e for e in entries if 'bc' not in e]
            unsortable.sort(key=lambda e: -e['i'])

            def cmp(a, b):
                if a['bc'] < b['bc']:
                    return -1
                if a['bc'] > b['bc']:
                    return 1
                return (a['i'] - b['i']) if not bias_right else (b['i'] - a['i'])
            sortable.sort(key=cmp_to_key(cmp))

            result = []
            idx = 0
            while unsortable and unsortable[-1]['i'] <= idx:
                result.append(unsortable.pop()['v'])
                idx += 1
            for e in sortable:
                result.append(e['v'])
                idx += 1
                while unsortable and unsortable[-1]['i'] <= idx:
                    result.append(unsortable.pop()['v'])
                    idx += 1
            return result

        best = [layer[:] for layer in layering]
        best_cc = cross_count(best)
        i = 0
        last_best = 0
        while last_best < 4 and i < 48:
            bias_right = (i % 4) >= 2
            if i % 2 == 0:
                for r in range(len(layering) - 2, -1, -1):
                    child_pos = {v: j for j, v in enumerate(layering[r + 1])}
                    layering[r] = sort_layer(layering[r], aug_out, child_pos, bias_right)
            else:
                for r in range(1, len(layering)):
                    parent_pos = {v: j for j, v in enumerate(layering[r - 1])}
                    layering[r] = sort_layer(layering[r], aug_in, parent_pos, bias_right)
            cc = cross_count(layering)
            if cc < best_cc:
                best_cc = cc
                best = [layer[:] for layer in layering]
                last_best = 0
            else:
                last_best += 1
            i += 1

        return self._dagre_transpose(best, aug_in, aug_out)

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
            if nodes[b]['dummy']:
                continue
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



