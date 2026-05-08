import os
import sys
import re
from collections import defaultdict, deque
import yaml
import pathlib
from types import SimpleNamespace

import onnx
import onnxslim
from onnxslim.utils import summarize_model, print_model_info_as_table


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

yolo_config_path = os.path.join(current_path, 'config/yolo26s_[320,640]_cfg.yaml')
yolo_pose_config_path = os.path.join(current_path, 'config/yolo26s-pose_[320,640]_cfg.yaml')


yolo_onnx_path = os.path.join(current_path, 'models_convert/original/yolo26s.onnx')
yolo_onnx_output_path = os.path.join(current_path, 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')

yolo_pose_onnx_path = os.path.join(current_path, 'models_convert/original/yolo26s-pose.onnx')
yolo_pose_onnx_output_path = os.path.join(current_path, 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')


def export(yolo_type:str="yolo"):
    if yolo_type == "yolo":
        config_path = yolo_config_path
        YOLO_module = YOLO

    elif yolo_type == "yolo-pose":
        config_path = yolo_pose_config_path
        YOLO_module = YOLO

    else:
        raise ValueError("yolo_type must be 'yolo', 'yolo-pose'")


    config = load_config(config_path)

    with temporary_chdir(current_path):
        model = YOLO_module(config.model)
        model.export(**vars(config))


def prune_exclusive_branch(graph:onnx.GraphProto, target_name: str):
    """
    精准删除目标节点及其独占上游父节点，并安全处理下游汇合点（如 G->D）。
    
    :param graph: onnx.GraphProto 对象
    :param target_name: 目标节点名 或 输出Tensor名

    :return: 被删除的节点名集合
    """

    # 1. 确保节点有唯一名称
    for i, node in enumerate(graph.node):
        if not node.name:
            node.name = f"auto_node_{i}"

    # 2. 构建映射表
    tensor_to_producer = {}
    tensor_to_consumers = defaultdict(list)
    for node in graph.node:
        for out in node.output:
            tensor_to_producer[out] = node
        for inp in node.input:
            tensor_to_consumers[inp].append(node)

    # 定位目标节点（支持传节点名或Tensor名）
    target_node = next((n for n in graph.node if n.name == target_name), None)
    if not target_node and target_name in tensor_to_producer:
        target_node = tensor_to_producer[target_name]
        print(f"从目标点'{target_name}'开始")

    if not target_node:
        raise ValueError(f"未找到目标节点或Tensor: '{target_name}'")

    # 3. 向上BFS：精准收集“独占上游”节点
    nodes_to_delete = set()
    queue = deque([target_node])
    
    while queue:
        node = queue.popleft()
        nodes_to_delete.add(node.name)

        for inp in node.input:
            if inp not in tensor_to_producer:
                continue  # 遇到 graph.input 或 initializer 停止
            
            parent = tensor_to_producer[inp]
            if parent.name in nodes_to_delete:
                continue

            # 核心判断：父节点是否被其他非删除分支共享？
            is_shared = False
            for out_t in parent.output:
                for consumer in tensor_to_consumers.get(out_t, []):
                    if consumer.name != node.name and consumer.name not in nodes_to_delete:
                        is_shared = True
                        break
                if is_shared:
                    break
            
            if not is_shared:
                queue.append(parent)

    nodes_to_delete.add(target_name)
    print(f"准备删除独占节点: {nodes_to_delete}")

    # 4. 处理下游汇合边
    deleted_tensors = set()
    for node in graph.node:
        if node.name in nodes_to_delete:
            deleted_tensors.update(node.output)

    for t_name in deleted_tensors:
        for consumer in tensor_to_consumers.get(t_name, []):
            if consumer.name not in nodes_to_delete:
                # 直接移除该输入边
                new_inputs = [i for i in consumer.input if i != t_name]
                del consumer.input[:]
                consumer.input.extend(new_inputs)
                print(f"已断开边: {t_name} -> {consumer.name} (剩余输入数: {len(consumer.input)})")


    # 5. 执行节点删除
    new_nodes = [n for n in graph.node if n.name not in nodes_to_delete]
    graph.node.clear()
    graph.node.extend(new_nodes)

    new_outputs = [out for out in graph.output if out.name not in nodes_to_delete]
    graph.output.clear()
    graph.output.extend(new_outputs)

    # 6. 清理孤儿Tensor/Initializer/ValueInfo
    used_tensors = set()
    for n in graph.node:
        used_tensors.update(n.input)
        used_tensors.update(n.output)

    for out in graph.output:
        used_tensors.add(out.name)

    new_input = [inp for inp in graph.input if inp.name in used_tensors]
    graph.input.clear()
    graph.input.extend(new_input)


    new_value_info = [vi for vi in graph.value_info if vi.name in used_tensors]
    graph.value_info.clear()
    graph.value_info.extend(new_value_info)

    new_initializer = [init for init in graph.initializer if init.name in used_tensors]
    graph.initializer.clear()
    graph.initializer.extend(new_initializer)

    return nodes_to_delete

def modify(yolo_type:str="yolo"):
    if yolo_type == "yolo":
        onnx_model_path = yolo_onnx_path
        onnx_model_output_path = yolo_onnx_output_path
    elif yolo_type == "yolo-pose":
        onnx_model_path = yolo_pose_onnx_path
        onnx_model_output_path = yolo_pose_onnx_output_path
    else:
        raise ValueError("yolo_type must be 'yolo' or 'yolo-pose'")

    onnx_model = onnx.load_model(onnx_model_path)

    original_info = summarize_model(onnx_model, os.path.basename(onnx_model_path))
    
    onnx_model = onnxslim.slim(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)

    onnx.save_model(onnx_model, onnx_model_output_path)

    slimmed_info = summarize_model(onnx_model, os.path.basename(onnx_model_output_path))
    print_model_info_as_table([original_info, slimmed_info])


yolo_type = "yolo"
# yolo_type = "yolo-pose"

export(yolo_type)
modify(yolo_type)