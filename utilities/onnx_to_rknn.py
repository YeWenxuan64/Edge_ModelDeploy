import sys
import os
import shutil
from pathlib import Path

from rknn.api import RKNN


current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import temporary_chdir, clean_files_or_dirs, read_dataset_txt_to_list
from accuracy_debugger import RknnAccuracyDebugger



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
        self.tmp_model_path = None
		
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).resolve()
        else:
            self.dataset_path = None

        self.target_platform = target_platform
        if self.target_platform not in ['rk3588', 'rk3576', 'rk3566']:
            raise ValueError("target_platform must be 'rk3588' or 'rk3576' or 'rk3566'")

        # self.extra_optimize()
        self.quantized_algorithm = 'normal'
        self.compress_weight = False
        self.model_pruning = False
        self.flash_attention = False

        # self.do_hybrid_quantization()
        self.custom_hybrid = None
        
        #self.set_do_accuracy_analysis()
        self.accuracy_analysis_picture_list = None

        self.file_or_dir_to_clean = ["check0_base_optimize.onnx", "check1_fold_constant.onnx", "check2_correct_ops.onnx", "check3_fuse_ops.onnx"]

    def extra_optimize(self, quantized_algorithm:str='normal', compress_weight:bool=False, model_pruning:bool=False, flash_attention:bool=False):
        """
        Args:
            quantized_algorithm (str): The quantization algorithm to use. 
                - Options: 'normal' for min-max quantization, 'kl_divergence' for KL divergence-based or 'mmse' for minimum mean square error quantization.
                - Default is 'normal'.

            compress_weight (bool): Whether to compress model weights to reduce memory usage. 
                - Default is False.

            model_pruning (bool): Whether to apply model pruning to remove less important parameters. 
                - Default is False.

            flash_attention (bool): Whether to use flash attention mechanism for faster attention computation. 
                - Default is False.
        """

        if quantized_algorithm not in ['normal', 'kl_divergence', 'mmse']:
            raise ValueError("quantized_algorithm must be 'normal' or 'kl_divergence' or 'mmse'")
        
        self.quantized_algorithm = quantized_algorithm
        self.compress_weight = compress_weight
        self.model_pruning = model_pruning
        self.flash_attention = flash_attention

        print(f"[OnnxToRKNN] extra_optimize: quantized_algorithm={self.quantized_algorithm}, compress_weight={self.compress_weight}, model_pruning={self.model_pruning}, flash_attention={self.flash_attention}")

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

        print(f"[OnnxToRKNN] do_hybrid_quantization: custom_hybrid={self.custom_hybrid}")

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

        print(f"[OnnxToRKNN] set_do_accuracy_analysis: accuracy_analysis_picture_list={self.accuracy_analysis_picture_list}")


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
            dataset_path_list = read_dataset_txt_to_list(self.dataset_path)

            tmp_dataset_path = self.tmp_dir / self.dataset_path.name
            with open(tmp_dataset_path, 'w') as f:
                for paths in dataset_path_list:
                    f.write(' '.join(paths) + '\n')

            self.dataset_path = tmp_dataset_path
            self.file_or_dir_to_clean.append(self.dataset_path)

        # 复制 onnx 模型到 tmp 目录，转换在 tmp 目录内进行，避免污染原模型
        self.tmp_model_path = self.tmp_dir / self.model_path.name
        shutil.copy2(self.model_path, self.tmp_model_path)
        self.file_or_dir_to_clean.append(self.tmp_model_path)

        with temporary_chdir(self.tmp_dir):
            self.self_convert(mean_rgb, std_rgb)

        if self.accuracy_analysis_picture_list is not None:
            debugger = RknnAccuracyDebugger(self.tmp_dir, self.tmp_model_path)
            debugger.plot_accuracy_analysis()
            debugger.plot_network_analysis(show=True) # 带路径追踪的精度分析（Netron 风格网络图）

    def clean(self):
        clean_files_or_dirs([str(self.tmp_dir / name) for name in self.file_or_dir_to_clean])
        if debugger is not None:
            debugger.clean()


    def self_convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        rknn = RKNN(verbose=True)

        # Pre-process config
        print('--> Config model')
        rknn.config(mean_values=mean_rgb, std_values=std_rgb, quantized_algorithm=self.quantized_algorithm, target_platform=self.target_platform, 
                    compress_weight=self.compress_weight, model_pruning=self.model_pruning, enable_flash_attention=self.flash_attention)
        print('done')

        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(model=str(self.tmp_model_path))
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
                self.file_or_dir_to_clean.extend([model_input, data_input, model_quantization_cfg])

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
            self.file_or_dir_to_clean.append(str(self.tmp_dir / "snapshot"))

        # Release
        rknn.release()
        print('--> Released rknn')






if __name__ == '__main__':
    # accuracy_analysis test
    parent_dir = current_dir.parent

    # MODEL_PATH = 'avtrack_ModelDeploy/models_convert/onnx/avtrack_[[1,3,112,112][1,3,224,224]].onnx'
    # RKNN_MODEL = 'avtrack_ModelDeploy/models_convert/rknn/avtrack_i8[[1,112,112,3][1,224,224,3]].rknn'
    MODEL_PATH = 'retinaface_mobile_ModelDeploy/models_convert/onnx/RetinaFace_mobile_[1,3,320,320].onnx'
    RKNN_MODEL = 'retinaface_mobile_ModelDeploy/models_convert/rknn/RetinaFace_mobile_i8[1,320,320,3].rknn'
    DATASET_PATH = str(parent_dir / 'datasets/datasets.txt')


    TARGET_PLATFORM = 'rk3588'
    converter = OnnxToRKNN(MODEL_PATH, RKNN_MODEL, DATASET_PATH, TARGET_PLATFORM)

    # 测试文件已存在，不做转换
    # 图结构分析使用 tmp 目录下的 ONNX 模型副本（convert() 会复制到此）
    tmp_model_path = converter.tmp_dir / converter.model_path.name

    # 精度分析调试器：读取精度分析文件、plt 显示、解析图结构
    debugger = RknnAccuracyDebugger(converter.tmp_dir, tmp_model_path)

    # 直接读取精度分析数据并绘制
    # debugger.plot_accuracy_analysis()

    # 带路径追踪的精度分析（Netron 风格网络图，多输入 -> 多输出 排列组合路径）
    debugger.plot_network_analysis()