import os
import json
import shutil
import platform
import subprocess
import heapq
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import cv2
import onnx


try:
    from utils import temporary_chdir

except ImportError:
    try:
        from .utils import temporary_chdir

    except ImportError:
        import sys
        current_dir = Path(__file__).parent.resolve()
        sys.path.append(str(current_dir))
        from utils import temporary_chdir


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




class OnnxToQNN:
    def __init__(self, model_path:str, qnn_model_path:str, dataset_path:str):
        """
        Initialize the ONNX to QNN converter.

        Args:
            model_path (str): Path to the input ONNX model file that needs to be converted.

            qnn_model_path (str): Path where the converted QNN model file will be saved.

            dataset_path (str | None): Path to a text file containing paths to dataset images for quantization. 
                - The text file should contain one image path per line for single-input models, 
                or multiple image paths separated by spaces for multi-input models.
                - Default is None. no quantization will be performed.
        """

        self.model_path = Path(model_path).resolve()
        self.qnn_model_path = Path(qnn_model_path).resolve()

        self.dataset_path = dataset_path
        if dataset_path:
            self.dataset_path = Path(dataset_path).resolve()

        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径

        qairt_path = Path(current_dir).resolve() / 'qairt'
        version_dir = next(qairt_path.iterdir())# 获取qairt目录下的第一个子目录
        self.qnn_sdk_dir = version_dir

        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径
        self.tmp_onnx_path = self.tmp_dir / self.model_path.name

        self.file_or_dir_to_clean = []

        self.set_quantization_method()
        self.use_custom_alibration_data()

    def set_quantization_method(self, param_quant_method:str='percentile', act_quant_method:str='entropy', bitwidth:str='w8a8', bias_bitwidth:int=8):
        """
        Configure quantization parameters for the model.
        
        Args:
            param_quant_method (str): Quantization method for model parameters (weights).
                - Available options: 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'.
                - Default: 'percentile'.

            act_quant_method (str): Quantization method for activations.
                - Available options: 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'.
                - Default: 'entropy'.

            bitwidth (str): Quantization bitwidth configuration in format 'w<W>a<A>', 
                where W is weight bitwidth and A is activation bitwidth.
                - Available options: 'w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16'.
                - Default: 'w8a8'.

            bias_bitwidth (int): Bitwidth for bias quantization.
                - Available options: 8, 32.
                - Default: 8.
        """

        if param_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('param_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if act_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('act_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')
        
        if bias_bitwidth not in [8, 32]:
            raise ValueError('bias_bitwidth must be 8 or 32')
        
        self.param_quant_method = param_quant_method
        self.act_quant_method = act_quant_method
        self.weights_bitwidth = int(bitwidth[1:2])  # 提取w后面的数字
        self.act_bitwidth = int(bitwidth[3:4])      # 提取a后面的数字
        self.bias_bitwidth = bias_bitwidth
        print(f"Quantization method has been set to param_quant_method={self.param_quant_method}, act_quant_method={self.act_quant_method}, bitwidth={self.weights_bitwidth}w{self.act_bitwidth}a")

    def use_custom_alibration_data(self, custom_alibration_data_path:str|None=None):
        """
        Args:
            custom_alibration_data_path (str | None): Path to a text file containing the custom calibration dataset.
                - Each line in the text file should represent a path to image data.
                - If the model has multiple inputs, the paths should be separated by spaces.
                
                - The calibration data must be preprocessed to match the model's input dimensions, format, and data type.
                - The data must be in .raw binary format generated by np.ndarray.tofile().
                
                - Example: If the model input is float32 data with shape [1, 3, 224, 224], 
                the images must be preprocessed to match this shape and data type before being converted to .raw format.
                ```
                resized_image = cv2.resize(image, (224, 224))
                tranposed_image = np.transpose(resized_image, (2, 0, 1))
                batched_image = np.expand_dims(tranposed_image, axis=0)
                np.float32(batched_image).tofile('image.raw')
                ```
        """

        if custom_alibration_data_path is None:
            self.custom_alibration_data_path = None
        else:
            self.custom_alibration_data_path = Path(custom_alibration_data_path).resolve()

    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]], set_input_order:str='nhwc'):
        """
        Args:
            mean_rgb (list[list[int | float,]], optional): Mean values for RGB channels normalization.
                - Each inner list contains 3 values (R, G, B) representing the mean for each channel in one input.
                - If multiple inputs are provided, For example, [[123, 116, 103], [123, 116, 103]]
                - Defaults to [[0, 0, 0]] (no mean normalization).
                
            std_rgb (list[list[int | float,]], optional): Standard deviation values for RGB channels normalization.
                - Each inner list should contain 3 values (R, G, B) representing the standard deviation for each channel in one input.
                - Similar to mean_rgb, can provide multiple lists for multiple inputs.
                - Defaults to [[1, 1, 1]] (no standard deviation normalization).

            set_input_order (str, optional): Input order for the converted model. 'nhwc' or 'nchw'.
                - Defaults to 'nhwc'.
        """

        self.run_env_script()

        self.modify_onnx_model(mean_rgb, std_rgb)

        onnx_model_info = self.get_onnx_model_info()
        if onnx_model_info is None:
            exit(1)

        dlc_model_path = self.convert_onnx_model(onnx_model_info, set_input_order)
        if dlc_model_path is None:
            exit(1)

        calibration_data_index_path = None
        if self.custom_alibration_data_path is None:
            if self.dataset_path:
                calibration_data_index_path = self.generate_calibration_data(onnx_model_info, set_input_order)
        else:
            calibration_data_index_path = self.custom_alibration_data_path

        if calibration_data_index_path is not None:
            quantized_dlc_model_path = self.quantize_model(dlc_model_path, calibration_data_index_path)
        else:
            quantized_dlc_model_path = dlc_model_path

        if quantized_dlc_model_path is None:
            exit(1)


        config_path = self.write_config_file(dlc_model_path)

        self.generate_context_binary_model(quantized_dlc_model_path, config_path)

    def clean(self):
        file_count = 0
        dir_count = 0

        for file_or_dir in self.file_or_dir_to_clean:
            try:
                if os.path.isfile(file_or_dir):
                    os.remove(file_or_dir)
                    file_count += 1

                elif os.path.isdir(file_or_dir):
                    # 统计目录中的文件数量
                    for root, dirs, files in os.walk(file_or_dir):
                        file_count += len(files)
                        dir_count += len(dirs)
                    
                    shutil.rmtree(file_or_dir) # 删除目录
                    dir_count += 1  # 加上被删除的目录本身

            except Exception as e:
                print(f"failed to delete {file_or_dir} due to {e}")

        print(f"cleaned {file_count} files and {dir_count} dirs")


    @staticmethod
    def run_subprocess(command:str) -> int:
        executable = '/bin/bash'
        print(f"Running command: {command}")

        # 使用实时输出的方式执行命令
        process = subprocess.Popen(command, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                                universal_newlines=True, executable=executable, env=os.environ)
        
        # 实时打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取返回码
        return_code = process.poll()

        return return_code

    def run_env_script(self):
        # Linux/Unix系统使用bash脚本
        envsetup_script = self.qnn_sdk_dir / 'bin/envsetup.sh'
        command = f"source '{envsetup_script}' && env"
        executable = '/bin/bash'
        encoding = 'utf-8'
        print("Setting up QAIRT Linux environment...")
            
        # 执行脚本
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable=executable)
        
        stdout, stderr = proc.communicate() # 获取输出
        
        if proc.returncode != 0:
            print(f"Error executing script: {stderr.decode(encoding)}")
            return False
        
        # 解析环境变量
        for line in stdout.decode().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
        
    def modify_onnx_model(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        self.tmp_dir.mkdir(exist_ok=True) # 确保tmp目录存在

        if not self.model_path.exists():
            print(f"Error: ONNX file not found at {self.model_path}")
            return None
        
        model = onnx.load_model(str(self.model_path))

        # 检查是否需要归一化
        need_mean_normalization = False
        need_std_normalization = False
        
        # 检查均值是否不为0
        if not np.allclose(np.array(mean_rgb, np.float32), 0.0):
            need_mean_normalization = True
        
        # 检查标准差是否不为1
        if not np.allclose(np.array(std_rgb, np.float32), 1.0):
            need_std_normalization = True
        

        if need_mean_normalization or need_std_normalization:
            nodes_to_add = []
            initializers_to_add = []

            # 获取原始输入信息
            for index, input_node in enumerate(model.graph.input):
                input_name = input_node.name

                mean_sub_node_name = f"{input_name}_Normalization_Sub"
                std_div_node_name = f"{input_name}_Normalization_Div"

                input_mean = mean_rgb[index]
                input_std = std_rgb[index]
                mean = np.array(input_mean, np.float32).reshape(1, len(input_mean), 1, 1)
                std = np.array(input_std, np.float32).reshape(1, len(input_std), 1, 1)

                current_input = input_name
                # 添加减法节点
                if need_mean_normalization:
                    # 将mean转换为ONNX张量
                    mean_tensor = onnx.numpy_helper.from_array(mean, name=f"{input_name}_mean_tensor")
                    
                    # 创建减法节点：(input - mean)
                    sub_output = input_name + "_sub"
                    sub_node = onnx.helper.make_node(
                        'Sub',
                        inputs=[input_name, mean_tensor.name],
                        outputs=[sub_output],
                        name=mean_sub_node_name
                    )

                    nodes_to_add.append(sub_node)
                    initializers_to_add.append(mean_tensor)

                    current_input = sub_output

                # 添加除法节点
                if need_std_normalization:
                    # 将std转换为ONNX张量
                    std_tensor = onnx.numpy_helper.from_array(std, name=f"{input_name}_std_tensor")
                    
                    # 创建除法节点：input / std
                    div_output = f"{input_name}_normalized"
                    div_node = onnx.helper.make_node(
                        'Div',
                        inputs=[current_input, std_tensor.name],
                        outputs=[div_output],
                        name=std_div_node_name
                    )

                    nodes_to_add.append(div_node)
                    initializers_to_add.append(std_tensor)

                    current_input = div_output


                # 更新所有使用原始输入的节点
                for node in model.graph.node:
                    if node.name != mean_sub_node_name and node.name != std_div_node_name:
                        for i, node_input in enumerate(node.input):
                            if node_input == input_name:
                                node.input[i] = current_input


            nodes_to_add.reverse()
            initializers_to_add.reverse()

            for node in nodes_to_add:
                model.graph.node.insert(0, node)
            for initializer in initializers_to_add:
                model.graph.initializer.insert(0, initializer)


        model = reorder_onnx_nodes_by_input(model, 5)
        model = reorder_onnx_nodes_by_output(model, 10)

        onnx.checker.check_model(model, full_check=True)
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


        # 复制ONNX文件到tmp目录
        onnx.save_model(model, str(self.tmp_onnx_path))
        self.file_or_dir_to_clean.append(self.tmp_onnx_path)
        print(f"Copied ONNX file to {self.tmp_onnx_path}")

    def get_onnx_model_info(self) -> dict|None:
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
            model = onnx.load_model(str(self.tmp_onnx_path))
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
        print(f"Model info: {model_info}")

        return model_info

    def convert_onnx_model(self, onnx_model_info:dict, set_input_order:str) -> str|None:

        layout_params = [] # 构建输入布局参数
        for input_info in onnx_model_info.get("inputs"): 
            input_name = input_info["name"]
            
            if set_input_order == 'nhwc': # 为每个输入添加源布局和目标布局参数
                layout_params.extend([f'--source_model_input_layout "{input_name}" NCHW', f'--desired_input_layout "{input_name}" NHWC'])
                
            layout_params.extend([f'--desired_input_color_encoding "{input_name}" rgb rgb'])
        
        layout_args = " ".join(layout_params) # 将布局参数拼接成字符串


        extra_args = '--target_backend HTP --onnx_summary' # --preserve_onnx_output_order

        if not self.dataset_path and not self.custom_alibration_data_path:
            extra_args += " --float_bitwidth 16"


        command = f"qairt-converter --input_network {str(self.tmp_onnx_path)} {layout_args} {extra_args}"

        return_code = self.run_subprocess(command)
        
        if return_code == 0:
            print("Convert onnx to qnn-dlc successful!")

            dlc_path = self.tmp_onnx_path.with_suffix('.dlc')
            self.file_or_dir_to_clean.append(dlc_path)
            return dlc_path
        
        else:
            return None
        
    def generate_calibration_data(self, onnx_model_info:dict, set_input_order:str) -> str|None:
        """
        生成校准数据
        
        Args:
            onnx_model_info (dict): 模型信息，包含输入尺寸
        
        Returns:
            list[str] | None: 校准数据文件路径列表，每个输入对应一个文件
        """
        try:
            # 读取数据集文件
            dataset_dir = str(self.dataset_path.parent)
            dataset_path_list = []

            with open(str(self.dataset_path), 'r') as f:
                lines = f.readlines() # 逐行读取文件

                for line in lines: # 如果行不为空，则分割路径
                    line = line.strip() # 去除首尾空白字符
                    if line:
                        one_line_paths_list = [path for path in line.split(' ') if path] # 按空格分割路径，并过滤掉空字符串

                        full_path_list = []
                        for img_path in one_line_paths_list:
                            # 将字符串转换为 Path 对象
                            p = Path(img_path)
                            
                            # 判断是否为绝对路径
                            if p.is_absolute():
                                # 如果已经是绝对路径，直接使用
                                full_path = p
                            else:
                                # 如果是相对路径，则与 dataset_dir 拼接
                                full_path = dataset_dir / p
                            
                            full_path_list.append(str(full_path))
                        
                        dataset_path_list.append(full_path_list)


            # 为每个输入创建目录和文件列表
            calibration_files = []
            for idx, input_info in enumerate(onnx_model_info["inputs"]):
                # 创建输出目录
                output_dir = self.tmp_dir / f"calibration_data_for_input{idx + 1}"
                output_dir.mkdir(parents=True, exist_ok=True)
                self.file_or_dir_to_clean.append(output_dir)

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

                    # 获取原始尺寸
                    orig_h, orig_w = img.shape[:2]

                    # 计算缩放比例
                    scale = min(width/orig_w, height/orig_h)
                    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

                    # 等比缩放
                    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                    # 创建目标尺寸的画布并居中放置图片
                    y_offset = (height - new_h) // 2
                    x_offset = (width - new_w) // 2
                    x1_pad = x_offset
                    x2_pad = width - (x_offset + new_w)
                    y1_pad = y_offset
                    y2_pad = height - (y_offset + new_h)

                    padded_image = cv2.copyMakeBorder(img_resized, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_REFLECT)

                    cv2.imshow("padded_image", padded_image)
                    cv2.waitKey(1)

                    # BGR转RGB
                    padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
                    padded_image = np.expand_dims(padded_image, axis=0)

                    if set_input_order == 'nchw':
                        # 转换为CHW格式
                        padded_image = np.transpose(padded_image, (0, 3, 1, 2)).copy(order='C')

                    # 转换为float32
                    img_float = padded_image.astype(np.float32)

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
            calibration_data_index = self.tmp_dir / f"calibration_data.txt"
            with open(str(calibration_data_index), 'w') as f:
                # 使用zip_longest处理不等长列表，空值用空字符串填充
                for row in zip_longest(*calibration_files, fillvalue=''):
                    # 过滤掉空字符串，但保留位置（这样列对齐）
                    formatted_row = ' '.join(item if item else '' for item in row)
                    f.write(formatted_row + '\n')
                print(f'{calibration_data_index} created listing {len(calibration_files)} columns.')

            print("Calibration data generation completed successfully!")

            self.file_or_dir_to_clean.append(calibration_data_index)
            for file_list in calibration_files:
                self.file_or_dir_to_clean.extend(file_list)

            return calibration_data_index

        except Exception as e:
            print(f"Error generating calibration data: {str(e)}")
            return None

    def quantize_model(self, dlc_model_path:str, calibration_data_index_path:str) -> str|None:
        dlc_model_file = Path(str(dlc_model_path))
        input_list_str = str(calibration_data_index_path)
        quantized_dlc_model_path = dlc_model_file.parent / f"{dlc_model_file.stem}_quantized.dlc"

        if not dlc_model_file.exists():
            print(f"Error: DLC model not found at {dlc_model_file}")
            return False
        

        quantize_args = f'--weights_bitwidth {self.weights_bitwidth}'
        quantize_args += f' --act_bitwidth {self.act_bitwidth} '
        quantize_args += f' --bias_bitwidth {self.bias_bitwidth}'
        quantize_args += f' --use_per_channel_quantization'
        quantize_args += f' --param_quantizer_calibration {self.param_quant_method}'
        quantize_args += f' --act_quantizer_calibration {self.act_quant_method}'

        extra_args = f'{quantize_args} --target_backend HTP'
        
        command = f"qairt-quantizer --input_dlc {dlc_model_path} --input_list {input_list_str} --output_dlc {quantized_dlc_model_path} {extra_args}"

        with temporary_chdir(self.tmp_dir):
            return_code = self.run_subprocess(command)

        self.file_or_dir_to_clean.append(self.tmp_dir / 'output')

        if return_code == 0:
            self.file_or_dir_to_clean.append(quantized_dlc_model_path)
            print("Model quantization completed successfully!")
            return quantized_dlc_model_path
        else:
            print("Error during model quantization.")
            return None

    def write_config_file(self, dlc_model_path:str) -> str:
        dlc_model_file = Path(str(dlc_model_path))

        config_backend_path = dlc_model_file.parent / "config_backend.json"
        config_file_path = dlc_model_file.parent / "config_file.json"

        # 创建配置字典
        config_backend = {
            "graphs": [
                {
                    "graph_names": [self.tmp_onnx_path.stem],  # 获取不带后缀的文件名
                    "vtcm_mb": 0
                }
            ],
            "devices": [
                {
                    "dsp_arch": "v68",
                    "soc_id": 35
                }
            ]
        }


        config_file = {
            "backend_extensions": {
                "shared_library_path": "libQnnHtpNetRunExtensions.so",
                "config_file_path": str(config_backend_path)
            }
        }

        # 将配置写入JSON文件
        with open(str(config_backend_path), 'w') as f:
            json.dump(config_backend, f, indent=4)  # indent=4 使输出格式化，更易读

        with open(str(config_file_path), 'w') as f:
            json.dump(config_file, f, indent=4)

        self.file_or_dir_to_clean.append(str(config_backend_path))
        self.file_or_dir_to_clean.append(str(config_file_path))
        
        print(f"Config file created at: {config_backend_path}")
        return config_file_path

    def generate_context_binary_model(self, quantized_dlc_model_path:str, config_path:str):

        command = f"qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHtp.so --config_file {config_path}"
        command += f" --dlc_path {quantized_dlc_model_path} --output_dir {self.qnn_model_path.parent} --binary_file {self.qnn_model_path.stem}"

        return_code = self.run_subprocess(command)

        if return_code == 0:
            print("Context binary generation completed successfully!")
            return True
        else:
            print("Error during context binary generation.")
            return False



if __name__ == "__main__":
    onnx_path = './yolo11s.onnx'
    qnn_model_path = './yolo11s.bin'
    dataset_path = './datasets/datasets_face.txt'
    
    mean_rgb = [[0, 0, 0]]
    std_rgb = [[255, 255, 255]]

    onnx_to_qnn = OnnxToQNN(onnx_path, qnn_model_path, dataset_path)
    onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.clean()


  
