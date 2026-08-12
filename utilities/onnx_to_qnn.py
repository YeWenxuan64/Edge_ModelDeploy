import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path
from itertools import zip_longest
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import cv2
import onnx



current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import temporary_chdir, reorder_onnx_nodes_by_input, reorder_onnx_nodes_by_output, sanitize_name, letterbox_image
from qnn_accuracy_debugger import SnpeAccuracyDebugger




class QnnHybridQuantGen:
    def __init__(self, custom_hybrid:list[list[str]], weights_bitwidth:int=8, act_bitwidth:int=16, bias_bitwidth:int=8, float_bitwidth:int|None=None):
        if float_bitwidth is not None:
            if float_bitwidth not in (16, 32):
                raise ValueError('float_bitwidth must be 16 or 32')
            dtype = 'float'
            weights_bitwidth = float_bitwidth
            act_bitwidth = float_bitwidth
            bias_bitwidth = float_bitwidth
        else:
            dtype = 'int'
            if weights_bitwidth not in (4, 8, 16):
                raise ValueError('weights_bitwidth must be 4, 8 or 16')
            if act_bitwidth not in (8, 16):
                raise ValueError('act_bitwidth must be 8 or 16')
            if bias_bitwidth not in (8, 32):
                raise ValueError('bias_bitwidth must be 8 or 32')

        if not isinstance(custom_hybrid, list) or not custom_hybrid:
            raise ValueError('custom_hybrid must be a non-empty list of [input_tensor, output_tensor] pairs')

        self.hybrid_quantization = {
            "custom_hybrid": custom_hybrid,
            "dtype": dtype,
            "weights_bitwidth": weights_bitwidth,
            "act_bitwidth": act_bitwidth,
            "bias_bitwidth": bias_bitwidth,
        }
        print(f"Hybrid quantization is set: {len(custom_hybrid)} subgraph(s) dtype={dtype} "
              f"w{weights_bitwidth}a{act_bitwidth}b{bias_bitwidth}, rest quantized by global settings")

    def generate_hybrid_quantization_overrides(self, tmp_onnx_path:str) -> str|None:
        """
        根据子图的输入/输出张量生成 QAIRT 混合量化的 quantization_overrides JSON。
        每个 [输入张量, 输出张量] 对之间的所有节点按 do_hybrid_quantization 指定的
        精度标记，转换器会在子图边界自动插入 Convert 节点。

        Returns:
            str | None: overrides JSON 文件路径；失败返回 None。
        """
        try:
            model = onnx.load_model(tmp_onnx_path)
        except Exception as e:
            print(f"Error loading ONNX model for hybrid quantization: {e}")
            return None

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
        try:
            for pair in self.hybrid_quantization["custom_hybrid"]:
                if len(pair) != 2:
                    raise ValueError(f"Each custom_hybrid pair must be [input_tensor, output_tensor], got {pair}")
                in_tensor = resolve_tensor(pair[0])
                out_tensor = resolve_tensor(pair[1])
                sub_middle = downstream_of(in_tensor) & upstream_of(out_tensor)
                if not sub_middle:
                    print(f"Warning: no nodes found between '{in_tensor}' and '{out_tensor}', skipped")
                middle |= sub_middle
        except ValueError as e:
            print(f"Error: {e}")
            return None

        if not middle:
            print("Error: no nodes selected for hybrid quantization")
            return None

        hq = self.hybrid_quantization
        dtype = hq["dtype"]
        act_bw = hq["act_bitwidth"]
        weight_bw = hq["weights_bitwidth"]
        bias_bw = hq["bias_bitwidth"]

        initializer_names = {init.name for init in model.graph.initializer}
        activation_encodings: dict[str, list[dict]] = {}
        param_encodings: dict[str, list[dict]] = {}

        # 识别 bias: 作为 Conv/ConvTranspose/Gemm 第3个输入(index 2)的 initializer
        bias_names = set()
        for n in nodes:
            if n.op_type in ('Conv', 'ConvTranspose', 'Gemm') and len(n.input) > 2:
                bias_names.add(n.input[2])

        for idx in middle:
            n = nodes[idx]
            for out in n.output:
                if out and out not in activation_encodings:
                    activation_encodings[out] = [{"bitwidth": act_bw, "dtype": dtype}]
            for inp in n.input:
                if inp in initializer_names and inp not in param_encodings:
                    param_bw = bias_bw if inp in bias_names else weight_bw
                    param_encodings[inp] = [{"bitwidth": param_bw, "dtype": dtype}]

        overrides = {
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
            "version": "0.5.0",
        }

        overrides_path = Path(tmp_onnx_path).parent / "quantization_overrides.json"
        with open(str(overrides_path), 'w') as f:
            json.dump(overrides, f, indent=4)

        

        print(f"Hybrid quantization overrides generated: {len(middle)} nodes, "
              f"{len(activation_encodings)} activation tensors, {len(param_encodings)} param tensors -> {overrides_path}")
        return str(overrides_path)


class OnnxToQNN:
    def __init__(self, model_path:str, qnn_model_path:str, dataset_path:str|None=None, target_platform:str='qcs6490'):
        """
        Initialize the ONNX to QNN converter.

        Args:
            model_path (str): Path to the input ONNX model file that needs to be converted.

            qnn_model_path (str): Path where the converted QNN model file will be saved.

            dataset_path (str | None): Path to a text file containing paths to dataset images for quantization. 
                - The text file should contain one image path per line for single-input models, 
                or multiple image paths separated by spaces for multi-input models.
                - Default is None. no quantization will be performed.

            target_platform (str): Target platform for the QNN model.
                - Available options: 'qcs6490', 'qcs8550', 'qcs9075'.
                - Default: 'qcs6490'.
        """

        self.model_path = Path(model_path).resolve()
        self.qnn_model_path = Path(qnn_model_path).resolve()

        self.dataset_path = dataset_path
        if dataset_path:
            self.dataset_path = Path(dataset_path).resolve()

        self.target_platform = target_platform
        self.architecture_dict = {
            "qcs6490": {"dsp_arch": "v68", "soc_id": 35},
            "qcs8550": {"dsp_arch": "v73", "soc_id": 66},
            "qcs9075": {"dsp_arch": "v73", "soc_id": 77},
        }

        if target_platform not in self.architecture_dict.keys():
            raise ValueError(f"Invalid target platform: {target_platform}. Available options: {self.architecture_dict.keys()}")


        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径

        qairt_path = Path(current_dir).resolve() / 'qairt'
        version_dir = next(qairt_path.iterdir())# 获取qairt目录下的第一个子目录
        self.qnn_sdk_dir = version_dir

        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径

        sanitize_model_name = sanitize_name(self.model_path.stem)
        tmp_onnx_path = self.tmp_dir / sanitize_model_name
        self.tmp_onnx_path = tmp_onnx_path.with_suffix('.onnx')

        self.file_or_dir_to_clean = []
        self.accuracy_analyzer = None
        self.hybrid_quantizer = None

        self.set_quantization_method()
        self.use_custom_alibration_data()

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
        bw_match = re.match(r'w(\d+)a(\d+)', bitwidth)
        self.weights_bitwidth = int(bw_match.group(1))  # 提取w后面的完整数字
        self.act_bitwidth = int(bw_match.group(2))  
        self.bias_bitwidth = bias_bitwidth
        print(f"Quantization method has been set to param_quant_method={self.param_quant_method}, act_quant_method={self.act_quant_method}, bitwidth={self.weights_bitwidth}w{self.act_bitwidth}a")

    def do_hybrid_quantization(self, custom_hybrid:list[list[str]], weights_bitwidth:int=8, act_bitwidth:int=16, bias_bitwidth:int=8, float_bitwidth:int|None=None):
        """
        设置混合量化(与 onnx_to_rknn.py 的 do_hybrid_quantization 一致)：
        通过子图的输入张量与输出张量指定区域，自动识别两者之间的所有节点，
        对这些节点使用指定精度，子图之外的节点仍按 set_quantization_method
        的全局设置(默认 w8a8)量化为 INT8。

        两种模式(二选一)：
        1. 整数混合量化(默认)：weights_bitwidth / act_bitwidth / bias_bitwidth
           指定区域内权重 / 激活 / 偏置的整数位宽。
           例如全局 w8a8、区域 w8a16：weights_bitwidth=8, act_bitwidth=16。
        2. 浮点保留：float_bitwidth 指定区域保持浮点精度，16=FP16，32=FP32。

        Args:
            custom_hybrid (list[list[str]]): 每个内层列表为 [输入张量名, 输出张量名]，
                表示一个混合量化子图：输入张量与输出张量之间的所有节点被选中。
                可传入多个子图，例如 [[in1, out1], [in2, out2]]。
                张量名也可以是节点名(自动取该节点的输出张量作为边界)。
            weights_bitwidth (int): 整数模式下区域内的权重位宽，可选 4/8/16。默认 8。
            act_bitwidth (int): 整数模式下区域内的激活位宽，可选 8/16。默认 16。
            bias_bitwidth (int): 整数模式下区域内的偏置位宽，可选 8/32。默认 8。
            float_bitwidth (int | None): 若设置(16/32)，区域保持浮点(FP16/FP32)，
                此时忽略三个整数位宽参数。默认 None 表示使用整数混合量化。
        """

        self.hybrid_quantizer = QnnHybridQuantGen(custom_hybrid, weights_bitwidth, act_bitwidth, bias_bitwidth, float_bitwidth)

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]|None=None):
        """
        Args:
            accuracy_analysis_picture_list (list[str], optional): A list of image paths required for model accuracy analysis. 
                - Each element in the list should be a path to an image. 
                - For models with a single input, provide a single image path. 
                - For models with multiple inputs, provide multiple image paths. Example: ['/home/xxx/1.jpg', '/home/xxx/2.jpg']
                - Defaults to None.
        """

        self.accuracy_analyzer = SnpeAccuracyDebugger(self.tmp_dir, self.tmp_onnx_path, accuracy_analysis_picture_list, self.run_subprocess)


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

        onnx_model_info = self.get_onnx_model_info(self.tmp_onnx_path)
        if onnx_model_info is None:
            exit(1)

        quantization_overrides_path = None
        golden_dlc_path = None
        if self.hybrid_quantizer is not None:
            quantization_overrides_path = self.hybrid_quantizer.generate_hybrid_quantization_overrides(self.tmp_onnx_path)
            if quantization_overrides_path is None:
                exit(1)

            # 混合量化时，精度分析的 golden 参考必须使用纯浮点 DLC：
            # 带混合量化编码(如 16-bit)的未量化 DLC 无法在 x86 CPU 的 --stage converted 阶段执行
            # (报 "No backend could validate")，会导致 golden 输出无法生成、精度分析失败。
            golden_dlc_path = self.convert_onnx_model(onnx_model_info, set_input_order, None, output_dlc_name=f"{self.tmp_onnx_path.stem}_golden")


        dlc_model_path = self.convert_onnx_model(onnx_model_info, set_input_order, quantization_overrides_path)
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

        if self.accuracy_analyzer is not None and quantized_dlc_model_path is not None:
            golden_dlc_path = golden_dlc_path or dlc_model_path
            self.accuracy_analyzer.set_model_inof(onnx_model_info, golden_dlc_path, quantized_dlc_model_path)

            return_code = self.accuracy_analyzer.accuracy_analysis(mean_rgb, std_rgb, set_input_order)

            self.file_or_dir_to_clean.append(str(self.accuracy_analyzer.working_dir))
            self.file_or_dir_to_clean.append(str(self.tmp_dir / "working_directory"))

            if return_code == 0:
                print("Accuracy analysis completed successfully.")
            else:
                print("Accuracy analysis failed.")

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


        model = reorder_onnx_nodes_by_input(model, 5)
        model = reorder_onnx_nodes_by_output(model, 10)

        onnx.checker.check_model(model, full_check=True)
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


        # 复制ONNX文件到tmp目录
        onnx.save_model(model, str(self.tmp_onnx_path))
        self.file_or_dir_to_clean.append(self.tmp_onnx_path)
        print(f"Copied ONNX file to {self.tmp_onnx_path}")


    @staticmethod
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
        print(f"Model info: {model_info}")

        return model_info

    def convert_onnx_model(self, onnx_model_info:dict, set_input_order:str, quantization_overrides_path:str|None=None, output_dlc_name:str|None=None) -> str|None:
        layout_params = [] # 构建输入布局参数
        for input_info in onnx_model_info.get("inputs"): 
            input_name = input_info["name"]
            
            if set_input_order == 'nhwc': # 为每个输入添加源布局和目标布局参数
                layout_params.extend([f'--source_model_input_layout "{input_name}" NCHW', f'--desired_input_layout "{input_name}" NHWC'])
                
            layout_params.extend([f'--desired_input_color_encoding "{input_name}" rgb rgb'])
        
        layout_args = " ".join(layout_params) # 将布局参数拼接成字符串


        extra_args = "--target_backend HTP --onnx_skip_simplification" # --onnx_summary' # --preserve_onnx_output_order

        if not self.dataset_path and not self.custom_alibration_data_path:
            extra_args += " --float_bitwidth 16"

        if quantization_overrides_path:
            self.file_or_dir_to_clean.append(quantization_overrides_path)

            extra_args += f" --quantization_overrides {quantization_overrides_path}"

            if self.hybrid_quantizer is not None:
                hybrid_quantization_dict = self.hybrid_quantizer.hybrid_quantization
                if hybrid_quantization_dict["dtype"] == "float":
                    extra_args += f" --float_bitwidth {hybrid_quantization_dict['weights_bitwidth']}"


        if output_dlc_name is not None:
            dlc_path = self.tmp_onnx_path.parent / f"{output_dlc_name}.dlc"
        else:
            dlc_path = self.tmp_onnx_path.with_suffix('.dlc')

        command = f"qairt-converter --input_network {str(self.tmp_onnx_path)} {layout_args} {extra_args} -o {dlc_path}"

        return_code = self.run_subprocess(command)
        
        if return_code == 0:
            print("Convert onnx to qnn-dlc successful!")

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

                    # 等比缩放 + 居中填充 + BGR转RGB + 布局/类型转换 (复用 utils.letterbox_image)
                    img_float = letterbox_image(
                        img,
                        (width, height),
                        output_format=('nchw' if set_input_order == 'nchw' else 'nhwc'),
                        output_dtype='float32',
                    )

                    # 调试窗口: 显示处理后的图像 (RGB -> BGR 保持颜色正确)
                    display_img = img_float.squeeze().astype(np.uint8)
                    if set_input_order == 'nhwc':
                        display_img = display_img
                    else:
                        display_img = np.transpose(display_img, (1, 2, 0))
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
        quantize_args += " --algorithms cle"

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

        architecture_config = self.architecture_dict[self.target_platform]

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
                    "dsp_arch": architecture_config["dsp_arch"],
                    "soc_id": architecture_config["soc_id"],
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
    onnx_to_qnn.set_quantization_method(param_quant_method='percentile', act_quant_method='entropy', bitwidth='w8a8', bias_bitwidth=8)

    # 可选: 混合量化 —— 与 onnx_to_rknn.py 的 do_hybrid_quantization 一致,
    # 通过子图输入/输出张量指定区域(自动识别两者之间的节点), 可指定多个子图,
    # 子图之外仍按全局 w8a8 量化。张量名可以是节点名(自动取该节点输出张量)。
    # 1) 整数混合量化: 区域 w8a16 (权重8bit, 激活16bit), 全局默认 w8a8
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], weights_bitwidth=8, act_bitwidth=16)
    # 2) 浮点保留: 区域保持 FP16
    # onnx_to_qnn.do_hybrid_quantization([['/model.0/conv/Conv', '/model.10/conv/Conv']], float_bitwidth=16)
    # 3) 多个子图
    # onnx_to_qnn.do_hybrid_quantization([['in1', 'out1'], ['in2', 'out2']], float_bitwidth=16)

    onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.clean()
