import os
import sys
import re
import yaml
import pathlib
from types import SimpleNamespace

import onnx
import onnxslim


current_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录的绝对路径
sys.path.insert(0, os.path.dirname(current_path))

project_root = os.path.join(current_path, 'models_convert/original/ultralytics')
sys.path.insert(0, project_root)
print(project_root)

# 定义一个上下文管理器以安全地更改目录
class temporary_chdir:
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = None
        
    def __enter__(self):
        self.saved_path = os.getcwd() # 保存进入前的当前目录
        os.chdir(self.new_path)       # 切换到新目录
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)     # 无论代码块是否报错，都恢复原来的目录

def load_config(config_path):

    def yaml_load(file, append_filename=False):
        """
        Load YAML data from a file.

        Args:
            file (str, optional): File name. Default is 'data.yaml'.
            append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

        Returns:
            (dict): YAML data and file name.
        """
        assert pathlib.Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()  # string

            # Remove special characters
            if not s.isprintable():
                s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

            # Add YAML filename to dict and return
            data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
            if append_filename:
                data["yaml_file"] = str(file)
            return data

    class IterableSimpleNamespace(SimpleNamespace):
        """
        An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

        This class extends the SimpleNamespace class with additional methods for iteration, string representation,
        and attribute access. It is designed to be used as a convenient container for storing and accessing
        configuration parameters.

        Methods:
            __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
            __str__: Returns a human-readable string representation of the object.
            __getattr__: Provides a custom attribute access error message with helpful information.
            get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

        Examples:
            >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
            >>> for k, v in cfg:
            ...     print(f"{k}: {v}")
            a: 1
            b: 2
            c: 3
            >>> print(cfg)
            a=1
            b=2
            c=3
            >>> cfg.get("b")
            2
            >>> cfg.get("d", "default")
            'default'

        Notes:
            This class is particularly useful for storing configuration parameters in a more accessible
            and iterable format compared to a standard dictionary.
        """

        def __iter__(self):
            """Return an iterator of key-value pairs from the namespace's attributes."""
            return iter(vars(self).items())

        def __str__(self):
            """Return a human-readable string representation of the object."""
            return "\n".join(f"{k}={v}" for k, v in vars(self).items())

        def __getattr__(self, attr):
            """Custom attribute access error message with helpful information."""
            name = self.__class__.__name__
            raise AttributeError(
                f"""
                '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
                'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
                {config_path} with the latest version from
                https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
                """
            )

        def get(self, key, default=None):
            """Return the value of the specified key if it exists; otherwise, return the default value."""
            return getattr(self, key, default)

    # Default configuration
    config_dict = yaml_load(config_path)
    for k, v in config_dict.items():
        if isinstance(v, str) and v.lower() == "none":
            config_dict[k] = None
    config = IterableSimpleNamespace(**config_dict)

    return config


try:
    from models_convert.original.ultralytics.ultralytics import YOLO
except Exception as e:
    from ultralytics import YOLO


yolo26_config_path = os.path.join(current_path, 'config/yolo26s_[320,640]_cfg.yaml')

config = load_config(yolo26_config_path)

with temporary_chdir(current_path):
    model = YOLO(config.model)
    model.export(**vars(config))