import os
import random
from pathlib import Path
import onnx  # 仅用于类型注解


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