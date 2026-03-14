import os
import json
import locale
import platform
import subprocess
import concurrent.futures
import shutil
from pathlib import Path
from itertools import zip_longest

import numpy as np
import cv2
import onnx




class OnnxToQNN:
    def __init__(self, model_path:str, qnn_model_path:str, dataset_path:str):
        self.platform = platform.system() # 'Windows' or 'Linux'

        self.model_path = Path(model_path).resolve()
        self.qnn_model_path = Path(qnn_model_path).resolve()
        self.dataset_path = Path(dataset_path).resolve()

        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径

        qairt_path = Path(current_dir).resolve() / 'qairt'
        version_dir = next(qairt_path.iterdir())# 获取qairt目录下的第一个子目录
        self.qnn_sdk_dir = version_dir

        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径
        self.tmp_onnx_path = self.tmp_dir / self.model_path.name


        self.set_debug_mode()
        self.set_quantization_method()
        self.custom_alibration_data_path:str|None = None
        self.accuracy_analysis_picture_path:str|None = None

    def set_debug_mode(self, debug_mode:bool=False,):
        self.debug_mode = debug_mode

    def set_quantization_method(self, param_quant_method:str='percentile', act_quant_method:str='entropy', bitwidth:str='w8a8', use_8bit_bias:bool=False):
        """
        重新配置量化参数
        
        参数:
            param_quant_method: 参数量化方法，可选 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'
            act_quant_method: 激活量化方法，可选 'min-max', 'sqnr', 'percentile', 'mse', 'entropy'
            bitwidth: 量化位数配置，格式为'wWaA'，其中w是权重位数，A是激活位数
                     可选 'w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16'
        """

        if param_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('param_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if act_quant_method not in ['min-max', 'sqnr', 'percentile', 'mse', 'entropy']:
            raise ValueError('act_quantization_method must be one of min-max, sqnr, percentile, mse, entropy')
        
        if bitwidth not in ['w4a8', 'w4a16', 'w8a8', 'w8a16', 'w16a16']:
            raise ValueError('bitwidth must be one of w4a8, w4a16, w8a8, w8a16, w16a16')
        
        self.param_quant_method = param_quant_method
        self.act_quant_method = act_quant_method
        self.weights_bitwidth = int(bitwidth[1:2])  # 提取w后面的数字
        self.act_bitwidth = int(bitwidth[3:4])      # 提取a后面的数字
        self.use_8bit_bias = use_8bit_bias
        print(f"Quantization method has been set to param_quant_method={self.param_quant_method}, act_quant_method={self.act_quant_method}, bitwidth={self.weights_bitwidth}w{self.act_bitwidth}a")

    def use_custom_alibration_data(self, custom_alibration_data_path:list[str]):
        self.custom_alibration_data_path = Path(custom_alibration_data_path).resolve()

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_path:str):
        self.accuracy_analysis_picture_path = os.path.abspath(accuracy_analysis_picture_path)

    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]], set_input_order:str='nhwc'):
        """
        Args:
            mean_rgb: Mean RGB values for each channel.
            std_rgb: Standard deviation RGB values for each channel.
            set_input_order: Input order for the model. 'nhwc' or 'nchw'.
        """

        self.run_env_script()

        onnx_model_info = self.get_onnx_model_info(mean_rgb, std_rgb)
        if onnx_model_info is None:
            exit(1)

        dlc_model_path = self.convert_onnx_model(onnx_model_info, set_input_order)
        if dlc_model_path is None:
            exit(1)

        if self.custom_alibration_data_path is None:
            calibration_data_index_path = self.generate_calibration_data(onnx_model_info, set_input_order)
        else:
            calibration_data_index_path = self.custom_alibration_data_path
        if calibration_data_index_path is None:
            exit(1)

        quantized_dlc_model_path = self.quantize_model(dlc_model_path, calibration_data_index_path)
        if quantized_dlc_model_path is None:
            exit(1)

        config_path = self.write_config_file(dlc_model_path)

        self.generate_context_binary_model(quantized_dlc_model_path, config_path)

        if self.debug_mode is False:
            if self.tmp_dir.exists():
                if Path('output').exists():
                    shutil.rmtree('output')
                    
                shutil.rmtree(self.tmp_dir)
                print(f"Temporary directory {self.tmp_dir} has been removed.")


    @staticmethod
    def run_subprocess(command:str) -> int:

        executable = '/bin/bash'
        print(f"Running command: {command}")

        # 使用实时输出的方式执行命令
        process = subprocess.Popen(command,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                                universal_newlines=True,
                                executable=executable, env=os.environ)
        
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
        envsetup_script = os.path.join(self.qnn_sdk_dir, 'bin/envsetup.sh')
        command = f"source '{envsetup_script}' && env"
        executable = '/bin/bash'
        encoding = 'utf-8'
        print("Setting up Linux environment...")
            

        # 执行脚本
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable=executable)
        
        # 获取输出
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            print(f"Error executing script: {stderr.decode(encoding)}")
            return False
        
        # 解析环境变量
        for line in stdout.decode().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                #print(f"Setting environment variable: {key}={value}")
                os.environ[key] = value
        
    def get_onnx_model_info(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]) -> dict|None:
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
        # 确保tmp目录存在
        self.tmp_dir.mkdir(exist_ok=True)

        # 清空tmp目录内容
        # for item in self.tmp_dir.iterdir():
        #     if item.is_file():
        #         item.unlink()
        #     elif item.is_dir():
        #         shutil.rmtree(item)

        # 复制ONNX文件到tmp目录
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
            mean_sub_node_name = "Normalization_Sub"
            std_div_node_name = "Normalization_Div"

            # 获取原始输入信息
            for index, input_node in enumerate(model.graph.input):
                input_name = input_node.name

                input_mean = mean_rgb[index]
                input_std = std_rgb[index]
                mean = np.array(input_mean, np.float32).reshape(1, len(input_mean), 1, 1)
                std = np.array(input_std, np.float32).reshape(1, len(input_std), 1, 1)


                # 添加减法节点
                if need_mean_normalization:
                    # 将mean转换为ONNX张量
                    mean_tensor = onnx.numpy_helper.from_array(mean, name="mean_tensor")
                    
                    # 创建减法节点：(input - mean)
                    sub_output = input_name + "_sub"
                    sub_node = onnx.helper.make_node(
                        'Sub',
                        inputs=[input_name, "mean_tensor"],
                        outputs=[sub_output],
                        name=mean_sub_node_name
                    )
                    model.graph.node.insert(0, sub_node)
                    model.graph.initializer.insert(0, mean_tensor)
                    current_input = sub_output
                else:
                    current_input = input_name


                # 添加除法节点
                if need_std_normalization:
                    div_output = input_name + "_normalized"

                    # 将std转换为ONNX张量
                    std_tensor = onnx.numpy_helper.from_array(std, name="std_tensor")
                    

                    # 创建除法节点：input / std
                    div_node = onnx.helper.make_node(
                        'Div',
                        inputs=[current_input, "std_tensor"],
                        outputs=[div_output],
                        name=std_div_node_name
                    )
                    insert_index = 0
                    if need_mean_normalization:
                        insert_index = 1
                    model.graph.node.insert(insert_index, div_node)
                    model.graph.initializer.insert(insert_index, std_tensor)

                    current_input = div_output


                # 更新所有使用原始输入的节点
                for node in model.graph.node:
                    # print(node.name)
                    if node.name != mean_sub_node_name and node.name != std_div_node_name:
                        for i, node_input in enumerate(node.input):
                            if node_input == input_name:
                                node.input[i] = current_input


            onnx.checker.check_model(model, full_check=True)
            model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        onnx.save_model(model, str(self.tmp_onnx_path))
        print(f"Copied ONNX file to {self.tmp_onnx_path}")


        try:
            # 加载ONNX模型
            model = onnx.load(str(self.tmp_onnx_path))
            
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

            return {"inputs": inputs, "outputs": outputs}
            
        except Exception as e:
            print(f"Error reading ONNX model: {str(e)}")
            return None

    def convert_onnx_model(self, onnx_model_info:dict, set_input_order:str) -> str|None:
        """
        转换ONNX模型
        
        Args:
            onnx_model_info: 包含模型输入输出信息的字典
        """

        layout_params = [] # 构建输入布局参数
        for input_info in onnx_model_info.get("inputs", []): 
            input_name = input_info["name"]
            
            if set_input_order == 'nhwc': # 为每个输入添加源布局和目标布局参数
                layout_params.extend([f'--source_model_input_layout "{input_name}" NCHW',
                                      f'--desired_input_layout "{input_name}" NHWC'])
                
            layout_params.extend([f'--desired_input_color_encoding "{input_name}" rgb rgb'])
        
        layout_args = " ".join(layout_params) # 将布局参数拼接成字符串
        
        extra_args = '--target_backend HTP --onnx_skip_simplification --onnx_summary'

        # 运行qairt-converter命令
        command = f"qairt-converter --input_network {str(self.tmp_onnx_path)} {layout_args} {extra_args}"

        return_code = self.run_subprocess(command)
        
        if return_code == 0:
            print("Conversion successful!")

            dlc_path = self.tmp_onnx_path.with_suffix('.dlc')
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
            with open(str(self.dataset_path), 'r') as f:
                image_path_list = [line.strip() for line in f if line.strip()]

            # 构建完整图片路径
            dataset_dir = str(self.dataset_path.parent)
            full_img_path_list = [os.path.join(dataset_dir, img_path) for img_path in image_path_list]

            # 为每个输入创建目录和文件列表
            calibration_files = []
            for idx, input_info in enumerate(onnx_model_info["inputs"]):
                # 创建输出目录
                output_dir = self.tmp_dir / f"calibration_data_for_input{idx + 1}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # 获取当前输入的尺寸
                input_shape = input_info["shape"]
                if len(input_shape) != 4 or input_shape[0] != 1:
                    print(f"Error: Unsupported input shape for input {idx + 1}")
                    continue

                height, width = input_shape[2], input_shape[3]

                def to_file_thread(img: np.ndarray, output_path: str):
                    img.tofile(output_path)
                    #print(f"Processed: {output_path}")

                max_workers = min(16, len(full_img_path_list))
                Threadpool_to_file = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                futures: list[concurrent.futures.Future] = []

                calibration_data_list = []

                # 处理每张图片
                for full_img_path in full_img_path_list:
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
            return calibration_data_index

        except Exception as e:
            print(f"Error generating calibration data: {str(e)}")
            return None

    def quantize_model(self, dlc_model_path:str, calibration_data_index_path:str) -> str|None:
        dlc_model_file = Path(str(dlc_model_path))

        if not dlc_model_file.exists():
            print(f"Error: DLC model not found at {dlc_model_file}")
            return False
        
        quantized_dlc_model_path = dlc_model_file.parent / "quantized_model.dlc"
        input_list_str = calibration_data_index_path

        if self.use_8bit_bias is True:
            bias_bitwidth = 8
        else:
            bias_bitwidth = 32


        quantize_args = f'--weights_bitwidth {self.weights_bitwidth} '
        quantize_args += f'--act_bitwidth {self.act_bitwidth} '
        quantize_args += f'--bias_bitwidth {bias_bitwidth} '
        quantize_args += f'--use_per_channel_quantization '
        quantize_args += f'--param_quantizer_calibration {self.param_quant_method} '
        quantize_args += f'--act_quantizer_calibration {self.act_quant_method} '


        extra_args = f'{quantize_args} --target_backend HTP'
        
        command = f"qairt-quantizer --input_dlc {dlc_model_path} --input_list {input_list_str} --output_dlc {quantized_dlc_model_path} {extra_args}"

        return_code = self.run_subprocess(command)

        if return_code == 0:
            print("Model quantization completed successfully!")
            return quantized_dlc_model_path
        else:
            print("Error during model quantization.")
            return None

    def write_config_file(self, dlc_model_path:str) -> str:
        dlc_model_file = Path(str(dlc_model_path))
        
        config_backend_path = dlc_model_file.parent / "config_backend.json"

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


        config_file_path = dlc_model_file.parent / "config_file.json"
        with open(str(config_file_path), 'w') as f:
            json.dump(config_file, f, indent=4)
        
        print(f"Config file created at: {config_backend_path}")
        return config_file_path

    def generate_context_binary_model(self, quantized_dlc_model_path:str, config_path:str):

        command = f"qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHtp.so \
            --dlc_path {quantized_dlc_model_path} \
            --output_dir {self.qnn_model_path.parent} \
            --binary_file {self.qnn_model_path.stem} \
            --config_file {config_path}"
        #  --profiling_level detailed
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
    # onnx_to_qnn.convert(mean_rgb, std_rgb)

    onnx_to_qnn.run_env_script()
  

    command = 'qairt-converter --input_network d:/Projects/IT/330Project/convert_models/yolo11s.onnx'
    command = ['python', 'D:/Projects/IT/330Project/convert_models/utilities/qairt/2.38.0.250901/bin/x86_64-windows-msvc/qairt-converter', '--input_network', 'd:/Projects/IT/330Project/convert_models/yolo11s.onnx']

    print(f"Running command: {command}")

    # 使用实时输出的方式执行命令
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=os.environ)
    
    # 实时打印输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())




    
