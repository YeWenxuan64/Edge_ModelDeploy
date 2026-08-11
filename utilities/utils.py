import os
import re
import random
import heapq
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


def collect_image_paths(dir_paths:list[str], max_count:int, random_sample:bool=False) -> str:
    """
    从多个图片目录中收集图片绝对路径，写入 tmp 目录下的 txt 文件，并返回该 txt 的绝对路径。
    
    :param dir_paths: 图片目录路径列表（支持相对或绝对路径）
    :param max_count: 总共最大读取图片数量
    :param random_sample: 是否随机取样。若为 True，且文件夹内图片数量大于 max_count 时，随机选取图片
    :return: 生成的 txt 文件的绝对路径字符串
    """
    if max_count <= 0:
        raise ValueError("max_count 必须大于 0")

    # 图片扩展名白名单（统一转为小写匹配）
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

    # 获取脚本所在目录的绝对路径
    try:
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        # 兼容 Jupyter / 交互式终端等没有 __file__ 的环境
        script_dir = Path.cwd().resolve()

    # 处理 tmp 目录
    tmp_dir = script_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

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
                str(file_path.resolve()) 
                for file_path in dir_path.iterdir() 
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS
            ]
            
            # 计算当前目录允许抽取的最大数量
            remaining_count = max_count - len(collected_paths)
            if remaining_count <= 0:
                break
                
            # 如果当前目录图片数大于剩余所需数量，随机抽取；否则全部加入
            if len(dir_images) > remaining_count:
                sampled_images = random.sample(dir_images, remaining_count)
                collected_paths.extend(sampled_images)
            else:
                collected_paths.extend(dir_images)
        else:
            # 顺序模式：保持原有逻辑，按顺序收集，达到 max_count 立即停止，节省性能
            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                    collected_paths.append(str(file_path.resolve()))
                    
                    if len(collected_paths) >= max_count:
                        break

        # 如果已收集够数量，跳出外层目录循环
        if len(collected_paths) >= max_count:
            break

    # 安全截断（防止随机抽样逻辑中因并发或计算误差多出元素）
    final_paths = collected_paths[:max_count]

    # 写入 txt 文件
    output_txt = tmp_dir / 'combind_image_paths.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_paths))
        if final_paths:
            f.write('\n')  # 符合 POSIX 文本规范，末尾保留换行

    return str(output_txt.resolve())


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


def reorder_onnx_nodes_by_input(model:onnx.ModelProto, max_depth:int=10) -> onnx.ModelProto:
    """
    按输入分支顺序重排 ONNX 节点，同时保证严格拓扑序
    核心策略: BFS分配优先级 + Kahn算法拓扑排序 + 最小堆调度
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

def reorder_onnx_nodes_by_output(model:onnx.ModelProto, max_depth:int=10) -> onnx.ModelProto:
    """
    按输出分支顺序重排 ONNX 节点 (output1优先 -> output2 -> ...)
    规则: output1 的祖先节点在前, output2 的在后，依此类推
    仅改变 model.graph.node 列表顺序，严格保证拓扑序以通过 full_check=True
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

def sanitize_name(name:str, replace_chars:str=r'()[]{}-\/:*?"<>|,') -> str:
    """将指定字符替换为 '_',并将连续下划线合并为一个"""
    trans_table = str.maketrans(replace_chars, '_' * len(replace_chars))
    name = name.translate(trans_table)
    name = re.sub(r'_+', '_', name).strip('_')
    return name



def generate_calibration_data(dataset_path:str, tmp_dir:str, onnx_model_info:dict, set_input_order:str='nhwc', process_singal_column:int=-1) -> str|None:
    """
    生成校准数据
    
    Args:
        onnx_model_info (dict): 模型信息，包含输入尺寸
    
    Returns:
        list[str] | None: 校准数据文件路径列表，每个输入对应一个文件
    """
    from itertools import zip_longest
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, Future

    dataset_path:Path = Path(dataset_path)
    tmp_dir:Path = Path(tmp_dir)
    file_or_dir_to_clean = []


    # 读取数据集文件
    dataset_dir = str(dataset_path.parent)
    dataset_path_list = []
    with open(str(dataset_path), 'r') as f:
        lines = f.readlines() # 逐行读取文件

        for line in lines: # 如果行不为空，则分割路径
            line = line.strip() # 去除首尾空白字符
            if line:
                one_line_paths_list = [path for path in line.split(' ') if path] # 按空格分割路径，并过滤掉空字符串
                full_path_list = []
                for img_path in one_line_paths_list:
                    p = Path(img_path) # 将字符串转换为 Path 对象
                    if p.is_absolute():
                        full_path = p # 如果已经是绝对路径，直接使用
                    else:
                        full_path = dataset_dir / p # 如果是相对路径，则与 dataset_dir 拼接
                    full_path_list.append(str(full_path))
                
                dataset_path_list.append(full_path_list)


    # 为每个输入创建目录和文件列表
    calibration_files = []
    for idx, input_info in enumerate(onnx_model_info["inputs"]):
        # 创建输出目录
        output_dir = tmp_dir / f"calibration_data_for_input{idx + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_or_dir_to_clean.append(output_dir)

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
                output_format=('nchw' if set_input_order == 'nchw' else 'nhwc'),
                output_dtype='float32',
            )

            # 调试窗口: 显示处理后的图像 (RGB -> BGR 保持颜色正确)
            if set_input_order == 'nhwc':
                display_img = img_float[0]
            else:
                display_img = np.transpose(img_float[0], (1, 2, 0))
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
    calibration_data_index = tmp_dir / f"calibration_data.txt"
    file_or_dir_to_clean.append(calibration_data_index)
    for file_list in calibration_files:
        file_or_dir_to_clean.extend(file_list)

    with open(str(calibration_data_index), 'w') as f:
        # 使用zip_longest处理不等长列表，空值用空字符串填充
        for row in zip_longest(*calibration_files, fillvalue=''):
            # 过滤掉空字符串，但保留位置（这样列对齐）
            formatted_row = ' '.join(item if item else '' for item in row)
            f.write(formatted_row + '\n')
        print(f'{calibration_data_index} created listing {len(calibration_files)} columns.')

    print("Calibration data generation completed successfully!")
    return calibration_data_index

