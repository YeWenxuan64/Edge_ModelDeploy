import os
from pathlib import Path
from rknn.api import RKNN

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




class OnnxToRKNN:
    def __init__(self, model_path:str, rknn_model_path:str, dataset_path:str|None=None, target_platform:str='rk3588'):
        """
        Initialize the ONNX to RKNN converter.

        Args:
            model_path (str): Path to the input ONNX model file that needs to be converted.

            rknn_model_path (str): Path where the converted RKNN model file will be saved.

            dataset_path (str | None): Path to a text file containing paths to dataset images for quantization. 
                - The text file should contain one image path per line for single-input models, 
                or multiple image paths separated by spaces for multi-input models.
                - Default is None. no quantization will be performed.

            target_platform (str): Target platform for the converted model. 
                - Supported platforms are 'rk3588', 'rk3576', 'rk3566'.
                - Defaults to 'rk3588'.
        """
        
        current_dir = Path(__file__).parent.resolve() # 获取当前文件所在目录的绝对路径
        self.tmp_dir = current_dir / 'tmp' # 构建tmp目录的绝对路径

        self.model_path = Path(model_path).resolve()
        self.rknn_model_path = Path(rknn_model_path).resolve()
		
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).resolve()
        else:
            self.dataset_path = None

        self.target_platform = target_platform
        if self.target_platform not in ['rk3588', 'rk3576', 'rk3566']:
            raise ValueError("target_platform must be 'rk3588' or 'rk3576' or 'rk3566'")

        self.extra_optimize()
        self.do_hybrid_quantization()
        self.set_do_accuracy_analysis()

        self.temp_files_list = ["check0_base_optimize.onnx", "check1_fold_constant.onnx", "check2_correct_ops.onnx", "check3_fuse_ops.onnx"]

    def extra_optimize(self, quantized_algorithm:str='kl_divergence', compress_weight:bool=False, model_pruning:bool=False, flash_attantion:bool=False):
        """
        Args:
            quantized_algorithm (str): The quantization algorithm to use. 
                - Options: 'normal' for basic quantization, 'kl_divergence' for KL divergence-based or 'mmse' for minimum mean square error quantization.
                - Default is 'kl_divergence'.

            compress_weight (bool): Whether to compress model weights to reduce memory usage. 
                - Default is False.

            model_pruning (bool): Whether to apply model pruning to remove less important parameters. 
                - Default is False.

            flash_attantion (bool): Whether to use flash attention mechanism for faster attention computation. 
                - Default is False.
        """

        if quantized_algorithm not in ['normal', 'kl_divergence', 'mmse']:
            raise ValueError("quantized_algorithm must be 'normal' or 'kl_divergence' or 'mmse'")
        
        self.quantized_algorithm = quantized_algorithm
        self.compress_weight = compress_weight
        self.model_pruning = model_pruning
        self.flash_attantion = flash_attantion

    def do_hybrid_quantization(self, custom_hybrid:list[list[str]]|None=None):
        """
        Args:
            custom_hybrid (list[list[str]], optional): A list of onnx node's input and output pair specifying the custom hybrid quantization settings.
                - Each inner list contains two strings representing the input name and output name of a subgraph in the ONNX model.
                - All nodes between the specified input and output will be quantized using FP16, 
                - while nodes outside these subgraphs will remain in 8-bit quantization. 
                - To apply hybrid quantization to multiple subgraphs, provide multiple pairs in the list, 
                e.g., [[input_name1, output_name1], [input_name2, output_name2]]. 
                - Defaults to None.
        """
        self.custom_hybrid = custom_hybrid

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]|None=None):
        """
        Args:
            accuracy_analysis_picture_list (list[str], optional): A list of image paths required for model accuracy analysis. 
                - Each element in the list should be a path to an image. 
                - For models with a single input, provide a single image path. 
                - For models with multiple inputs, provide multiple image paths. Example: ['/home/xxx/1.jpg', '/home/xxx/2.jpg']
                - Defaults to None.
        """
        if accuracy_analysis_picture_list is not None:
            self.accuracy_analysis_picture_list = [str(Path(path).resolve()) for path in accuracy_analysis_picture_list]
        else:
            self.accuracy_analysis_picture_list = None
		

    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
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
        """

        self.tmp_dir.mkdir(exist_ok=True)

        if self.dataset_path is not None: # 读取数据集文件
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

            tmp_dataset_path = self.tmp_dir / self.dataset_path.name
            with open(tmp_dataset_path, 'w') as f:
                for paths in dataset_path_list:
                    f.write(' '.join(paths) + '\n')

            self.dataset_path = tmp_dataset_path
            self.temp_files_list.append(self.dataset_path)

        with temporary_chdir(self.tmp_dir):
            self.self_convert(mean_rgb, std_rgb)

    def self_convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        rknn = RKNN(verbose=True)

        # Pre-process config
        print('--> Config model')
        rknn.config(mean_values=mean_rgb, std_values=std_rgb, quantized_algorithm=self.quantized_algorithm, target_platform=self.target_platform, 
                    compress_weight=self.compress_weight, model_pruning=self.model_pruning, enable_flash_attention=self.flash_attantion)
        print('done')

        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(model=str(self.model_path))
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')
        
        # Build model
        print('--> Building model')
        if self.dataset_path is not None:
            if self.custom_hybrid is None:
                ret = rknn.build(do_quantization=True, dataset=self.dataset_path)
            else:
                model_name = self.model_path.stem  # 获取文件名不带扩展名
                model_input = model_name + ".model" # 表示第一步生成的模型文件
                data_input = model_name + ".data" # 表示第一步生成的配置文件
                model_quantization_cfg = model_name + ".quantization.cfg" # 表示第一步生成的量化配置文件
                self.temp_files_list.extend([model_input, data_input, model_quantization_cfg])

                ret = rknn.hybrid_quantization_step1(dataset=self.dataset_path, proposal=False, custom_hybrid=self.custom_hybrid)
                ret = rknn.hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)  
        else:
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export rknn model')
        
        os.makedirs(self.rknn_model_path.parent, exist_ok=True)
        
        ret = rknn.export_rknn(str(self.rknn_model_path))
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        if self.accuracy_analysis_picture_list is not None:
            print(f'accuracy_analysis_picture_list: {self.accuracy_analysis_picture_list}')
            rknn.accuracy_analysis(inputs=self.accuracy_analysis_picture_list)

        # Release
        rknn.release()
        print('--> Released rknn')

    def clean(self):
        for file_name in self.temp_files_list:
            file_path = str(self.tmp_dir / file_name)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"deleted tmp file {file_path}")

            except Exception as e:
                print(f"failed to delete {file_path}: {str(e)}")