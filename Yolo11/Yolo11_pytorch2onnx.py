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

project_root = os.path.join(current_path, 'models_convert/original/ultralytics_yolo11')
sys.path.insert(0, project_root)


from ultralytics.engine import exporter


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


model_quantities = ['s', 'm']

yolo11_config_paths = [f'config/yolo11{size}_[320,640]_cfg.yaml' for size in model_quantities]
yolo11_config_paths = [os.path.join(current_path, path) for path in yolo11_config_paths]

model_paths = [f'models_convert/original/yolo11{size}.onnx' for size in model_quantities]
model_paths = [os.path.join(current_path, path) for path in model_paths]

output_paths = [f'models_convert/onnx/yolo11{size}_[1,3,320,640].onnx' for size in model_quantities]
output_paths = [os.path.join(current_path, path) for path in output_paths]


for yolo11_config_path, model_path, output_path in zip(yolo11_config_paths, model_paths, output_paths):

    config = load_config(yolo11_config_path)
    exporter.export(config)

    # 加载ONNX模型
    model = onnx.load(model_path)

    target_node_name_list = ['/model.23/cv2.0/cv2.0.2/Conv', '/model.23/cv2.1/cv2.1.2/Conv', '/model.23/cv2.2/cv2.2.2/Conv',
                             '/model.23/Sigmoid', '/model.23/Sigmoid_1', '/model.23/Sigmoid_2',
                             '/model.23/Clip', '/model.23/Clip_1', '/model.23/Clip_2']
    
    new_output_name_list = ['/model.23/cv2.0/cv2.0.2/Conv_output_0', '/model.23/cv2.1/cv2.1.2/Conv_output_0', '/model.23/cv2.2/cv2.2.2/Conv_output_0',
                            '/model.23/Sigmoid_output_0', '/model.23/Sigmoid_1_output_0', '/model.23/Sigmoid_2_output_0',
                            '/model.23/Clip_output_0', '/model.23/Clip_1_output_0', '/model.23/Clip_2_output_0']

    output_list = []
    for node in model.graph.node:
        for i, target_node_name in enumerate(target_node_name_list):
            if node.name == target_node_name:
                output_list.append((i, node.output[0]))
                node.output[0] = new_output_name_list[i]

    print(output_list)
    for node in model.graph.node:
        for (index, target_output_name) in output_list:
            if len(node.input) != 0 and node.input[0] == target_output_name:
                node.input[0] = new_output_name_list[index]

    for output in model.graph.output:
        for (index, target_output_name) in output_list:
            if output.name == target_output_name:
                output.name = new_output_name_list[index]

        print(output.name)

    print('start simplify')
    onnxslim.slim(model, output_path)

