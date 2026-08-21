import os
import re
import random
import heapq
import shutil
from pathlib import Path
from collections import defaultdict, deque
import onnx  # 仅用于类型注解


import os
import re
import random
import heapq
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import cv2
import onnx  # 仅用于类型注解
import onnxruntime as ort


# 一个上下文管理器以安全地更改目录
class temporary_chdir:
    def __init__(self, new_path:str):
        self.new_path = str(new_path)
        self.saved_path = None
        
    def __enter__(self):
        self.saved_path = os.getcwd() # 保存进入前的当前目录
        os.makedirs(self.new_path, exist_ok=True)  # 确保目录存在
        os.chdir(self.new_path)       # 切换到新目录
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)     # 无论代码块是否报错，都恢复原来的目录

class OnnxExecutor():
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.session = None
        self.providers = ['CPUExecutionProvider']

        self.input_names:list[str] = []
        self.output_names:list[str] = []
        self.float_inputs = False

        self.init_onnx()

    def init_onnx(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)

        input_details:list = self.session.get_inputs()
        self.input_names:list[str] = [inp.name for inp in input_details]
        self.output_names:list[str]  = [out.name for out in self.session.get_outputs()]

        if "float" in input_details[0].type:
            self.float_inputs = True
        else:
            self.float_inputs = False

    def get_input_shapes(self) -> list[tuple[int, int]]:
        input_details = self.session.get_inputs()

        input_sizes_list:list[tuple[int, int]] = []
        for input_detail in input_details:
            input_shape:list[int] = input_detail.shape
            input_sizes_list.append(tuple(input_shape))

        return input_sizes_list
        
    def put(self, input_data:list[np.ndarray], input_format:str='nhwc') -> list[np.ndarray]:
        if input_format == 'nhwc':
            input_data = [np.transpose(tensor, (0, 3, 1, 2)) for tensor in input_data]
        elif input_format == 'nchw':
            pass

        if self.float_inputs:
            input_data = [tensor.astype(np.float32) for tensor in input_data]

        input_feed = {} # 构建 feed_dict
        for i, input_name in enumerate(self.input_names):
            input_feed[input_name] = input_data[i]

        outputs = self.session.run(None, input_feed) # 执行推理
        return outputs
    
    def release(self):
        if self.session is not None:
            del self.session
            self.session = None

            self.input_names.clear()
            self.output_names.clear()

            print("ONNX Executor released")


def clean_files_or_dirs(file_or_dir_list:list) -> tuple[int, int]:
    """
    清理一组文件/目录（与 OnnxToQNN / OnnxToRKNN / AimetOnnxQuantizer 的 clean() 共用）。
    逐个尝试删除：
        - 文件 -> os.remove
        - 目录 -> 递归统计文件数/子目录数后 shutil.rmtree
    单项删除失败时打印告警但不中断，全部处理完后打印汇总。

    Args:
        file_or_dir_list (list): 待清理的文件/目录路径列表（Path 或 str 均可）。

    Returns:
        tuple[int, int]: (删除的文件数, 删除的目录数)。
    """
    file_count = 0
    dir_count = 0

    for file_or_dir in file_or_dir_list:
        file_or_dir = str(file_or_dir)
        try:
            if os.path.isfile(file_or_dir):
                os.remove(file_or_dir)
                file_count += 1

            elif os.path.isdir(file_or_dir):
                # 统计目录中的文件数量
                for root, dirs, files in os.walk(file_or_dir):
                    file_count += len(files)
                    dir_count += len(dirs)

                shutil.rmtree(file_or_dir)  # 删除目录
                dir_count += 1  # 加上被删除的目录本身

        except Exception as e:
            print(f"failed to delete {file_or_dir} due to {e}")

    print(f"cleaned {file_count} files and {dir_count} dirs")
    return file_count, dir_count

def sanitize_name(name:str, replace_chars:str=r'()[]{}-\/:*?"<>|,') -> str:
    """将指定字符替换为 '_',并将连续下划线合并为一个"""
    trans_table = str.maketrans(replace_chars, '_' * len(replace_chars))
    name = name.translate(trans_table)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def parse_bitwidth(bitwidth:str) -> tuple[int, int]:
    """
    Args:
        bitwidth (str): Bitwidth config in the format 'w<W>a<A>'.

    Returns:
        tuple[int, int]: (weights_bitwidth, act_bitwidth).
    """
    bw_match = re.match(r'w(\d+)a(\d+)', bitwidth)
    weights_bitwidth = int(bw_match.group(1))  # 提取w后面的完整数字
    act_bitwidth = int(bw_match.group(2))
    return weights_bitwidth, act_bitwidth

def letterbox_image(img:np.ndarray, target_shape:tuple[int, int], output_format:str='hwc', output_dtype:str='float32', border_value:tuple[int, int, int]|None=None) -> np.ndarray:
    """
    等比缩放并居中填充 (letterbox)，BGR -> RGB，并按指定布局/数据类型输出。

    Args:
        img (np.ndarray): OpenCV 读取的 BGR 图像。
        target_shape (tuple[int, int]): 目标尺寸，格式为 (宽, 高)。
        output_format (str): 输出布局。
            - 'hwc': [H, W, C]
            - 'chw': [C, H, W]
            - 'nhwc': [1, H, W, C]
            - 'nchw': [1, C, H, W]
            - Default: 'hwc'.
        output_dtype (str): 输出数据类型。
            - 'float32': Float32, 数值范围为原始像素 (0-255)。
            - 'uint8': Uint8, 数值范围为原始像素 (0-255)。
            - Default: 'float32'.
        border_value (tuple[int, int, int] | None): 填充像素值 (B,G,R)。
            - None (默认): 使用 cv2.BORDER_REFLECT 镜像填充。
            - 指定如 (114, 114, 114): 使用 cv2.BORDER_CONSTANT 纯色填充 (YOLO 惯例灰色)。

    Returns:
        np.ndarray: 处理后的 RGB 图像。
    """
    if output_format not in ['nchw', 'nhwc', 'chw', 'hwc']:
        raise ValueError("output_format must be one of 'nchw', 'nhwc', 'chw', 'hwc'")

    if output_dtype not in ['float32', 'uint8']:
        raise ValueError("output_dtype must be 'float32' or 'uint8'")

    width, height = target_shape  # 元组解包: (宽, 高)

    orig_h, orig_w = img.shape[:2]

    # 计算缩放比例
    scale = min(width / orig_w, height / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    # 等比缩放
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 创建目标尺寸的画布并居中放置图片
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    x1_pad = x_offset
    x2_pad = max(0, width - (x_offset + new_w))
    y1_pad = y_offset
    y2_pad = max(0, height - (y_offset + new_h))

    if border_value is None:
        padded_image = cv2.copyMakeBorder(img_resized, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_REFLECT)
    else:
        padded_image = cv2.copyMakeBorder(img_resized, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=border_value)

    # BGR转RGB
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

    # 按指定布局排列
    if output_format == 'chw':
        rgb_image = np.transpose(rgb_image, (2, 0, 1))              # [C, H, W]
    elif output_format == 'nhwc':
        rgb_image = rgb_image[None, ...]                            # [1, H, W, C]
    elif output_format == 'nchw':
        rgb_image = np.transpose(rgb_image, (2, 0, 1))[None, ...]   # [1, C, H, W]
    # 'hwc': 保持 [H, W, C]

    # 按指定数据类型输出
    if output_dtype == 'float32':
        return rgb_image.astype(np.float32)
    return rgb_image.astype(np.uint8)

def collect_image_paths(dir_paths:list[str], max_count:int=0, random_sample:bool=False, step:int=1, use_relative:bool=False) -> str:
    """
    从多个图片目录中收集图片路径，写入 txt 文件，并返回该 txt 的绝对路径。
    
    :param dir_paths: 图片目录路径列表（支持相对或绝对路径）
    :param max_count: 总共最大读取图片数量。0 表示不限制（收集全部）；负数会抛 ValueError
    :param random_sample: 是否随机取样。若为 True，且文件夹内图片数量大于 max_count 时，随机选取图片
    :param step: 间隔挑选步长（仅顺序模式生效）。step=1 表示全选；step=2 表示每隔 1 张取 1 张（
        即取第 0、2、4... 张）；step=N 表示每 N 张取 1 张（取第 0、N、2N... 张）。
    :param use_relative: 是否以相对于当前工作目录的形式输出图片路径。True 时，位于当前目录下的
        图片写成 './path/to/image.jpg' 格式；位于当前目录之外的图片写成 '../xxx' 相对形式。
        默认 False，输出绝对路径。
    :return: 生成的 txt 文件的绝对路径字符串
    """
    if max_count < 0:
        raise ValueError("max_count 不能为负数")
    if step <= 0:
        raise ValueError("step 必须大于 0")

    # max_count=0 表示不限制数量，收集全部；否则为数量上限
    limit = max_count if max_count > 0 else None

    # 图片扩展名白名单（统一转为小写匹配）
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

    collected_paths: list[str] = []

    for dir_str in dir_paths:
        dir_path = Path(dir_str).resolve()
        if not dir_path.is_dir():
            print(f"[警告] 路径不存在或不是目录，已跳过: {dir_path}")
            continue

        # 遍历当前目录下的文件（默认非递归。如需包含子目录，将 iterdir() 改为 rglob('*')）
        if random_sample:
            # 随机取样模式：需要先收集当前目录下的所有图片，以便进行等概率随机抽样
            dir_images = [
                file_path.resolve().as_posix()
                for file_path in dir_path.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS
            ]
            
            # 计算当前目录允许抽取的最大数量（max_count=0 表示不限制，全部纳入）
            remaining_count = None if limit is None else limit - len(collected_paths)
            if remaining_count is not None and remaining_count <= 0:
                break
                
            # 如果当前目录图片数大于剩余所需数量，随机抽取；否则全部加入
            if remaining_count is not None and len(dir_images) > remaining_count:
                sampled_images = random.sample(dir_images, remaining_count)
                collected_paths.extend(sampled_images)
            else:
                collected_paths.extend(dir_images)
        else:
            # 顺序模式：保持原有逻辑，按顺序收集，达到 max_count 立即停止，节省性能
            img_index = 0  # 当前目录内已扫描的图片计数（用于间隔挑选）
            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                    # 间隔挑选：step>1 时，只保留 img_index 为 step 整数倍的图片（第 0、step、2*step... 张）
                    if step > 1 and img_index % step != 0:
                        img_index += 1
                        continue
                    img_index += 1
                    collected_paths.append(file_path.resolve().as_posix())
                    
                    if limit is not None and len(collected_paths) >= limit:
                        break

        # 如果已收集够数量，跳出外层目录循环
        if limit is not None and len(collected_paths) >= limit:
            break

    # 安全截断（防止随机抽样逻辑中因并发或计算误差多出元素）；max_count=0 时不截断
    final_paths = collected_paths if limit is None else collected_paths[:limit]

    # 若 use_relative=True，将绝对路径转换为相对于当前工作目录的路径（./xxx 或 ../xxx 格式）
    if use_relative:
        cwd = Path.cwd()
        rel_paths = []
        for p in final_paths:
            rel = os.path.relpath(p, cwd).replace(os.sep, '/')
            if not rel.startswith('..'):  # 位于当前目录下时，使用 ./ 前缀
                rel = './' + rel
            # 位于当前目录之外时保留 ../ 相对形式（不额外加 ./）
            rel_paths.append(rel)
        final_paths = rel_paths

    # 写入 txt 文件（直接写在脚本运行/当前工作目录，不再生成 tmp 文件夹）
    output_txt = Path.cwd() / 'combind_image_paths.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_paths))
        if final_paths:
            f.write('\n')  # 符合 POSIX 文本规范，末尾保留换行

    return str(output_txt.resolve())

def read_dataset_txt_to_list(dataset_txt_path: str) -> list[list[str]]:
    txt_path = Path(dataset_txt_path).resolve()

    # 读取数据集文件
    dataset_dir = txt_path.parent
    dataset_path_list:list[list[str]] = []

    with open(str(txt_path), 'r') as f:
        lines = f.readlines() # 逐行读取文件

        for line in lines: # 如果行不为空，则分割路径
            line = line.strip() # 去除首尾空白字符
            if line:
                one_line_paths_list = [path for path in line.split(' ') if path] # 按空格分割路径，并过滤掉空字符串

                one_line_path_list = []
                for img_path in one_line_paths_list:
                    p = Path(img_path)
                    
                    if p.is_absolute():
                        full_path = p # 如果已经是绝对路径，直接使用
                    else:
                        full_path = dataset_dir / p # 如果是相对路径，则与 dataset_dir 拼接
                    
                    one_line_path_list.append(str(full_path))
                
                dataset_path_list.append(one_line_path_list)

    return dataset_path_list


def read_txt_line(txt_path: str, line_index: int = 0) -> list[str]:
    """
    Read a specified line of a text file, split it by whitespace, and
    convert each token into a full path string.

    - Relative paths are resolved against the directory containing the txt file.
    - Absolute paths are kept as-is.
    - Returns an empty list if the target line is empty.

    Args:
        txt_path (str): Path to the text file (e.g. a quantization dataset list).
        line_index (int): The 0-based line number to read. Default 0 (first line).
            Negative values count from the end (-1 = last line), mirroring Python
            list indexing. Raises IndexError if the index is out of range.

    Returns:
        list[str]: Full path strings parsed from the target line.

    Raises:
        FileNotFoundError: If txt_path does not exist.
        IndexError: If line_index is out of range.
    """
    txt_path:Path = Path(txt_path).resolve()

    if not txt_path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {txt_path}")

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    try:
        target_line = lines[line_index]
    except IndexError:
        raise IndexError(
            f"行索引 {line_index} 超出范围，文件共 {len(lines)} 行: {txt_path}"
        )

    # 按任意空白字符（空格/Tab/多空格）分割目标行，并过滤掉空字符串
    one_line_paths = target_line.split()

    full_path_list = []
    for img_path in one_line_paths:
        p = Path(img_path)
        if p.is_absolute():  # 已经是绝对路径，直接使用
            full_path = p
        else:  # 相对路径则与 txt 文件所在目录拼接
            full_path = txt_path.parent / p
        full_path_list.append(str(full_path))

    return full_path_list

def fmt_model_name_with_shape(model:onnx.ModelProto|str, model_name:str, use_nhwc:bool=False) -> str:
    """
    Parse ONNX model input shapes and format them into a descriptive model name string.

    Args:
        model:      Loaded onnx.ModelProto object (obtained via onnx.load()).
        model_name: Model name prefix, e.g. "yolo_", "track_".
        use_nhwc:   Whether to output shapes in NHWC order. Default False = NCHW order.

    Returns:
        Formatted model name string with `{shapes}` replaced. Examples:
            "yolo_{{shapes}}"              → "yolo_[1,3,160,320]"
            "yolo_{{shapes}}.onnx"          → "yolo_[1,3,160,320].onnx"
            "track_{{shapes}}"             → "track_[[1,3,128,128][1,3,256,256]]"
    """
    if isinstance(model, str):
        model = onnx.load_model(model)

    shapes = []
    for inp in model.graph.input:
        shape_dims = []
        for dim in inp.type.tensor_type.shape.dim:
            val = dim.dim_value if dim.dim_value > 0 else -1
            shape_dims.append(val)
        shapes.append(shape_dims)

    # 若 use_nhwc 且为 4 维形状，重排 (N,C,H,W) → (N,H,W,C)
    if use_nhwc and len(shapes[0]) == 4:
        shapes = [
            [s[0], s[2], s[3], s[1]] if len(s) == 4 else s
            for s in shapes
        ]

    if len(shapes) == 1:
        shape_str = str(shapes[0]).replace(" ", "")
    else:
        inner = "".join(str(s) for s in shapes)
        shape_str = f"[{inner}]".replace(" ", "")

    return model_name.format(shapes=shape_str)


def reorder_onnx_nodes_by_input(model:onnx.ModelProto, max_depth:int=10, aggressive:bool=False) -> onnx.ModelProto:
    """
    按输入分支顺序重排 ONNX 节点，同时保证严格拓扑序
    核心策略: BFS分配优先级 + Kahn算法拓扑排序 + 最小堆调度

    Args:
        max_depth (int): BFS 从各输入向下的最大搜索深度。
        aggressive (bool): 激进模式（默认 False）。True 时会把"仅依赖 initializer 的
            前驱节点"（如 QDQ 模型的 weight_dq 反量化节点）的优先级按下游消费节点
            所属输入分支反向回传，使 Kahn 排序中同优先级节点按输入顺序 tie-break。
            用于 QDQ 模型：AIMET 导出的 QDQ 里各输入的 Q -> DQ -> weight_dq -> 首个
            真实算子链可能被排乱，导致 qairt-converter 推导的 DLC 输入顺序与
            graph.input 不一致；aggressive=True 可修复。纯 float 模型无 weight_dq，
            该模式不影响其结果。
    """
    graph = model.graph

    # 1. 获取真实输入（排除 initializer 常量）
    init_names = {init.name for init in graph.initializer}

    # 兼容 sparse_initializer (部分量化模型会使用)
    if hasattr(graph, 'sparse_initializer'):
        init_names.update({init.values.name for init in graph.sparse_initializer if init.values.name})
        
    input_names = [inp.name for inp in graph.input if inp.name not in init_names]

    if len(input_names) < 2:
        print("less than 2 inputs, skip reorder")
        return model
    
    # for node in graph.node[:10]:
    #     print(node.name)

    nodes_list = list(graph.node) # 物化 graph.node，解决 Protobuf 迭代导致 id() 不稳定的问题
    max_depth = min(max_depth, len(nodes_list))

    # 2. 构建图依赖映射
    tensor_to_producer = {}          # tensor_name -> 生产它的 node
    tensor_to_consumers = defaultdict(list) # tensor_name -> [消费它的 node, ...]
    for node in nodes_list:
        for out in node.output:
            tensor_to_producer[out] = node
        for inp in node.input:
            tensor_to_consumers[inp].append(node)

    # 3. 多源 BFS 分配优先级 (depth, input_index)
    # 优先级规则：深度越小越靠前；同深度时，input_index 越小越靠前
    node_priority = {}
    bfs_queue = deque()
    visited_tensors = set(input_names) | init_names

    for idx, inp_name in enumerate(input_names):
        bfs_queue.append((inp_name, 0, idx))  # (tensor_name, current_depth, origin_input_idx)

    while bfs_queue:
        tensor, depth, inp_idx = bfs_queue.popleft()
        if depth >= max_depth:
            continue

        for node in tensor_to_consumers.get(tensor, []):
            nid = id(node)
            node_depth = depth + 1
            new_prio = (node_depth, inp_idx)
            
            # 记录最优优先级（更浅深度 或 更靠前的输入分支）
            if nid not in node_priority or new_prio < node_priority[nid]:
                node_priority[nid] = new_prio
                for out in node.output:
                    if out not in visited_tensors:
                        visited_tensors.add(out)
                        bfs_queue.append((out, node_depth, inp_idx))

    if aggressive:
        # 激进模式：把优先级沿输入边反向回传（只取更优：更浅深度 / 更靠前的输入分支）。
        # QDQ 模型中首个真实算子的权重反量化节点(weight_dq)只消费 initializer，正向
        # BFS 永远访问不到，只能拿默认优先级并按原始插入顺序 tie-break，导致输入分支
        # 顺序错乱。反向回传让 weight_dq 继承其下游消费节点所属输入分支的序号，
        # 从而各输入消费链按 graph.input 顺序排列。
        changed = True
        while changed:
            changed = False
            for node in nodes_list:
                prio = node_priority.get(id(node))
                if prio is None:
                    continue
                depth, inp_idx = prio
                for inp in node.input:
                    producer = tensor_to_producer.get(inp)
                    if producer is None:
                        continue
                    pid = id(producer)
                    new_prio = (max(depth - 1, 0), inp_idx)
                    old = node_priority.get(pid)
                    if old is None or new_prio < old:
                        node_priority[pid] = new_prio
                        changed = True

    # 未访问到的节点（超过深度或独立分支）赋予最低优先级
    default_prio = (max_depth + 1, len(input_names))
    for node in nodes_list:
        if id(node) not in node_priority:
            node_priority[id(node)] = default_prio

    # 4. 基于优先级的拓扑排序 (Kahn算法 + 最小堆)
    in_degree = {id(n): 0 for n in nodes_list}
    node_to_consumers = defaultdict(list)

    for node in nodes_list:
        for inp in node.input:
            producer = tensor_to_producer.get(inp)
            # 仅统计由图中其他节点产生的输入（自动忽略 initializers 和 graph.input）
            if producer is not None:
                in_degree[id(node)] += 1
                node_to_consumers[id(producer)].append(id(node))

    # 初始化最小堆: (depth, input_idx, stable_counter, node_id)
    heap = []
    counter = 0
    for nid, deg in in_degree.items():
        if deg == 0:
            d, idx = node_priority[nid]
            heapq.heappush(heap, (d, idx, counter, nid))
            counter += 1

    sorted_nodes = []
    id_to_node = {id(n): n for n in nodes_list}

    while heap:
        _, _, _, curr_id = heapq.heappop(heap)
        sorted_nodes.append(id_to_node[curr_id])

        for cons_id in node_to_consumers[curr_id]:
            in_degree[cons_id] -= 1
            if in_degree[cons_id] == 0:
                d, idx = node_priority[cons_id]
                heapq.heappush(heap, (d, idx, counter, cons_id))
                counter += 1

    if len(sorted_nodes) != len(graph.node):
        raise RuntimeError("Topological sort failed: graph contains cycles or missing dependencies.")

    # 5. 替换 protobuf repeated field
    del graph.node[:]
    graph.node.extend(sorted_nodes)

    print("reorder nodes by input successfully")
    # for node in graph.node[:10]:
    #     print(node.name)

    return model

def reorder_onnx_nodes_by_output(model:onnx.ModelProto, max_depth:int=10, aggressive:bool=False) -> onnx.ModelProto:
    """
    按输出分支顺序重排 ONNX 节点 (output1优先 -> output2 -> ...)
    规则: output1 的祖先节点在前, output2 的在后，依此类推
    仅改变 model.graph.node 列表顺序，严格保证拓扑序以通过 full_check=True

    Args:
        max_depth (int): 反向 BFS 从各输出向上的最大搜索深度。
        aggressive (bool): 为与 reorder_onnx_nodes_by_input 保持接口一致而保留的
            开关，此处不改变行为（输入分支顺序由 input 版本决定，输出版本只负责
            保持深度节点的原始相对顺序）。
    """

    graph = model.graph
    output_names = [out.name for out in graph.output]
    if len(output_names) < 2:
        print("output node less than 2, skip reorder")
        return model

    # for node in graph.node[-25:]:
    #     print(node.name)

    
    nodes_list = list(graph.node) # 一次性物化 graph.node，解决 Protobuf 迭代导致 id() 不稳定的问题
    max_depth = min(max_depth, len(nodes_list))
    original_indices = {id(n): i for i, n in enumerate(nodes_list)}

    # 构建 tensor -> producer 映射
    tensor_to_producer = {}
    for node in nodes_list:
        for out in node.output:
            tensor_to_producer[out] = node

    # 1. 反向 BFS 分配优先级
    # output1 -> prio 0, output2 -> prio 1, ... 值越小越先出堆，越排在列表前面
    node_priority = {}
    bfs_queue = deque()
    visited_tensors = set(output_names)

    for idx, out_name in enumerate(output_names):
        bfs_queue.append((out_name, 0, idx))  # (tensor, depth, priority)

    while bfs_queue:
        tensor, depth, prio = bfs_queue.popleft()
        if depth >= max_depth:
            continue

        producer = tensor_to_producer.get(tensor)
        if producer is None:
            continue  # 追溯到 graph.input 或 initializer，自动停止

        nid = id(producer)
        # 共享节点归属优先级更高（值更小，即更靠近 output1）的分支
        if nid not in node_priority or prio < node_priority[nid]:
            node_priority[nid] = prio
            for inp in producer.input:
                if inp not in visited_tensors:
                    visited_tensors.add(inp)
                    bfs_queue.append((inp, depth + 1, prio))

    # 未访问到的节点（超过深度或早期公共层）赋予最高优先级(-1)
    # 确保它们被调度到列表最前面，保持原图自然流向，输出分支节点集中在列表末尾
    default_prio = -1
    for node in nodes_list:
        if id(node) not in node_priority:
            node_priority[id(node)] = default_prio

    # 2. 基于优先级的拓扑排序 (Kahn算法 + 最小堆)
    in_degree = {id(n): 0 for n in nodes_list}
    node_to_consumers = defaultdict(list)

    for node in nodes_list:
        for inp in node.input:
            producer = tensor_to_producer.get(inp)
            # 仅统计由图中其他节点产生的输入（自动忽略 initializers 和 graph.input）
            if producer is not None:
                pid = id(producer)
                if pid in in_degree:  # 防御性检查
                    in_degree[id(node)] += 1
                    node_to_consumers[pid].append(id(node))

    # 初始化最小堆: (branch_priority, original_index, stable_counter, node_id)
    heap = []
    counter = 0
    for nid, deg in in_degree.items():
        if deg == 0:
            prio = node_priority[nid]
            orig_idx = original_indices[nid]
            heapq.heappush(heap, (prio, orig_idx, counter, nid))
            counter += 1

    sorted_nodes = []
    id_to_node = {id(n): n for n in nodes_list}

    while heap:
        _, _, _, curr_id = heapq.heappop(heap)
        sorted_nodes.append(id_to_node[curr_id])

        for cons_id in node_to_consumers[curr_id]:
            in_degree[cons_id] -= 1
            if in_degree[cons_id] == 0:
                prio = node_priority[cons_id]
                orig_idx = original_indices[cons_id]
                heapq.heappush(heap, (prio, orig_idx, counter, cons_id))
                counter += 1

    if len(sorted_nodes) != len(nodes_list):
        raise RuntimeError("Topological sort failed: graph contains cycles or missing dependencies.")

    # 3. 替换 protobuf repeated field
    del graph.node[:]
    graph.node.extend(sorted_nodes)

    print("reorder nodes by output successfully")
    # for node in graph.node[-25:]:
    #     print(node.name)

    return model


def find_hybrid_subgraph_nodes(model:onnx.ModelProto, custom_hybrid:list[list[str]], warn_prefix:str='') -> list[int]:
    """
    识别混合量化子图节点：对每个 [输入张量, 输出张量] 对，找出"输入张量下游 ∩ 输出张量
    上游"的所有节点下标并取并集（按节点下标升序，保证确定性）。

    供 QAIRT quantization_overrides 生成（QnnHybridQuantGen.generate_hybrid_quantization_overrides）
    与 AIMET 混合精度张量查找（AimetOnnxQuantizer._find_subgraph_tensors）共用，
    避免两份几乎相同的 producer/consumers/downstream/upstream 搜索逻辑重复维护。

    Args:
        model (onnx.ModelProto): ONNX 模型（读取 graph.node / graph.input）。
        custom_hybrid (list[list[str]]): 每个内层列表为 [输入张量, 输出张量]；
            张量名也可以是节点名（自动取该节点第一个输出张量作为边界）。
        warn_prefix (str): 某个子图无节点时的警告前缀（如 '[AIMET] '）。

    Returns:
        list[int]: 子图节点下标列表（升序，保证确定性）。

    Raises:
        ValueError: pair 格式错误，或张量/节点名在模型中不存在。
    """
    nodes = list(model.graph.node)

    # 张量 -> 产生它的节点下标
    producer: dict[str, int] = {}
    for idx, n in enumerate(nodes):
        for out in n.output:
            producer[out] = idx

    # 张量 -> 消费它的节点下标列表
    consumers: dict[str, list[int]] = {}
    for idx, n in enumerate(nodes):
        for inp in n.input:
            consumers.setdefault(inp, []).append(idx)

    graph_input_names = {i.name for i in model.graph.input}

    def resolve_tensor(name: str) -> str:
        """张量名 -> 张量名；若为节点名则取其第一个输出张量作为边界"""
        if name in producer or name in graph_input_names:
            return name
        for n in nodes:
            if n.name == name:
                if not n.output:
                    raise ValueError(f"Node '{name}' has no output tensor")
                return n.output[0]
        raise ValueError(f"Tensor or node '{name}' not found in the model")

    def downstream_of(tensor: str) -> set[int]:
        """消费该张量(直接或间接)的节点集合"""
        result: set[int] = set()
        stack = list(consumers.get(tensor, []))
        while stack:
            idx = stack.pop()
            if idx in result:
                continue
            result.add(idx)
            for out in nodes[idx].output:
                stack.extend(consumers.get(out, []))
        return result

    def upstream_of(tensor: str) -> set[int]:
        """产生该张量(直接或间接)的节点集合"""
        result: set[int] = set()
        start = producer.get(tensor)
        if start is None:
            return result
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in result:
                continue
            result.add(idx)
            for inp in nodes[idx].input:
                p = producer.get(inp)
                if p is not None:
                    stack.append(p)
        return result

    middle: set[int] = set()
    for pair in custom_hybrid:
        if len(pair) != 2:
            raise ValueError(f"Each custom_hybrid pair must be [input_tensor, output_tensor], got {pair}")
        in_tensor = resolve_tensor(pair[0])
        out_tensor = resolve_tensor(pair[1])
        sub_middle = downstream_of(in_tensor) & upstream_of(out_tensor)
        if not sub_middle:
            print(f"{warn_prefix}Warning: no nodes found between '{in_tensor}' and '{out_tensor}', skipped")
        middle |= sub_middle

    return sorted(middle)


def get_onnx_model_info(tmp_onnx_path:str) -> dict|None:
    """
    获取ONNX模型的输入输出信息
    
    Args:
        onnx_path (str): ONNX模型文件路径
        
    Returns:
        Dict: 包含模型输入输出信息的字典，格式如下：
        {
            "inputs": [
                {
                    "name": str,
                    "shape": List[int]
                }
            ],
            "outputs": [
                {
                    "name": str,
                    "shape": List[int]
                }
            ]
        }
    """
    
    try:
        # 加载ONNX模型
        model = onnx.load_model(tmp_onnx_path)
    except Exception as e:
        print(f"Error reading ONNX model: {str(e)}")
        return None
    
    # 获取输入信息
    inputs = []
    for input in model.graph.input:
        input_info = {
            "name": input.name,
            "shape": [d.dim_value if d.dim_value != 0 else 'dynamic' for d in input.type.tensor_type.shape.dim]
        }
        inputs.append(input_info)

    # 获取输出信息
    outputs = []
    for output in model.graph.output:
        output_info = {
            "name": output.name,
            "shape": [d.dim_value if d.dim_value != 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]
        }
        outputs.append(output_info)

    model_info = {"inputs": inputs, "outputs": outputs}

    return model_info

def normalize_onnx_model(model:onnx.ModelProto, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]) -> onnx.ModelProto:
    """
    Args:
        model (onnx.ModelProto): 待归一化的 ONNX 模型
        mean_rgb (list[list[int | float]]): RGB 均值
        std_rgb (list[list[int | float]]): RGB 标准差

    Returns:
        onnx.ModelProto: 归一化后的 ONNX 模型
    """

        # 检查是否需要归一化
    need_mean_normalization = False
    need_std_normalization = False
    
    # 检查均值是否不为0
    for mean in mean_rgb:
        if not np.allclose(np.array(mean, np.float32).flatten(), 0.0):
            need_mean_normalization = True
            break
    
    # 检查标准差是否不为1
    for std in std_rgb:
        if not np.allclose(np.array(std, np.float32).flatten(), 1.0):
            need_std_normalization = True
    

    if need_mean_normalization or need_std_normalization:
        nodes_to_add = []
        initializers_to_add = []

        # 获取原始输入信息
        for index, input_node in enumerate(model.graph.input):
            input_name = input_node.name

            input_mean = np.array(mean_rgb[index], np.float32)
            input_std = np.array(std_rgb[index], np.float32)
            channel_num = len(input_mean)

            # 用 1x1 卷积实现归一化: y = (x - mean) / std = x * (1/std) - mean/std
            # 权重为对角矩阵 diag(1/std)，偏置为 -mean/std
            conv_weight = (np.diag(1.0 / input_std)).astype(np.float32).reshape(channel_num, channel_num, 1, 1)
            conv_bias = (-input_mean / input_std).astype(np.float32)

            weight_tensor = onnx.numpy_helper.from_array(conv_weight, name=f"{input_name}_norm_weight")
            bias_tensor = onnx.numpy_helper.from_array(conv_bias, name=f"{input_name}_norm_bias")

            conv_output = f"{input_name}_normalized"
            conv_node = onnx.helper.make_node(
                'Conv',
                inputs=[input_name, weight_tensor.name, bias_tensor.name],
                outputs=[conv_output],
                name=f"{input_name}_Normalization_Conv"
            )

            nodes_to_add.append(conv_node)
            initializers_to_add.append(weight_tensor)
            initializers_to_add.append(bias_tensor)

            # 更新所有使用原始输入的节点
            # (新节点尚未插入 graph，因此这里无需像原来那样跳过自身)
            for node in model.graph.node:
                for i, node_input in enumerate(node.input):
                    if node_input == input_name:
                        node.input[i] = conv_output


        nodes_to_add.reverse()
        initializers_to_add.reverse()

        for node in nodes_to_add:
            model.graph.node.insert(0, node)
        for initializer in initializers_to_add:
            model.graph.initializer.insert(0, initializer)


    return model



