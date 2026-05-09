import os
from pathlib import Path


# 一个上下文管理器以安全地更改目录
class temporary_chdir:
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = None
        
    def __enter__(self):
        self.saved_path = os.getcwd() # 保存进入前的当前目录
        os.chdir(self.new_path)       # 切换到新目录
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)     # 无论代码块是否报错，都恢复原来的目录






def collect_image_paths(dir_paths: list[str], max_count: int) -> str:
    """
    从多个图片目录中收集图片绝对路径，写入 tmp 目录下的 txt 文件，并返回该 txt 的绝对路径。
    
    :param dir_paths: 图片目录路径列表（支持相对或绝对路径）
    :param max_count: 总共最大读取图片数量
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
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                collected_paths.append(str(file_path.resolve()))
                
                if len(collected_paths) >= max_count:
                    break

        if len(collected_paths) >= max_count:
            break

    # 安全截断（防止最后一次 break 后多出元素）
    final_paths = collected_paths[:max_count]

    # 写入 txt 文件
    output_txt = tmp_dir / 'combind_image_paths.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_paths))
        if final_paths:
            f.write('\n')  # 符合 POSIX 文本规范，末尾保留换行

    return str(output_txt.resolve())