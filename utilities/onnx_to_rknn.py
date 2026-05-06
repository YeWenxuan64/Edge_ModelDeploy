import os
from rknn.api import RKNN


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


class OnnxToRKNN:
    def __init__(self, model_path:str, rknn_model_path:str, dataset_path:str|None=None, target_platform:str='rk3588'):
        """
        Args:
            model_path (str): onnx model to convert
            rknn_model_path (str): output rknn model path
            dataset_path (str): a dataset paths text file for accuracy analysis
            target_platform (str): target platform, only 'rk3588' and 'rk3576' are supported
        """

        self.model_path = os.path.abspath(model_path)
        self.rknn_model_path = os.path.abspath(rknn_model_path)

        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
        self.tmp_dir = os.path.join(current_dir, 'tmp') # 构建tmp目录的绝对路径
		
        if dataset_path is not None:
            self.dataset_path = os.path.abspath(dataset_path)
        else:
            self.dataset_path = None
        self.target_platform = target_platform

        self.extra_optimize()
        self.custom_hybrid = None
        self.accuracy_analysis_picture_list:list[str]|None = None

        self.temp_files_list = ["check0_base_optimize.onnx", "check1_fold_constant.onnx", "check2_correct_ops.onnx", "check3_fuse_ops.onnx"]

    def extra_optimize(self, quantized_algorithm:str='kl_divergence', compress_weight:bool=False, model_pruning:bool=False, flash_attantion:bool=False):
        """
        Args:
            quantized_algorithm (str): 'normal' or 'kl_divergence' or 'mmse'
        """
        self.quantized_algorithm = quantized_algorithm
        self.compress_weight = compress_weight
        self.model_pruning = model_pruning
        self.flash_attantion = flash_attantion


    def do_hybrid_quantization(self, custom_hybrid:list[list[str]]):
        self.custom_hybrid = custom_hybrid

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]):
        self.accuracy_analysis_picture_list = [os.path.abspath(path) for path in accuracy_analysis_picture_list]
		

    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
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
        ret = rknn.load_onnx(model=self.model_path)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')
        
        # Build model
        if self.dataset_path is not None:
            if self.custom_hybrid is None:
                ret = rknn.build(do_quantization=True, dataset=self.dataset_path)
            else:
                model_name = os.path.basename(self.model_path).replace('.onnx','')
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
        
        os.makedirs(os.path.dirname(self.rknn_model_path), exist_ok=True)
        
        ret = rknn.export_rknn(self.rknn_model_path)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        if self.accuracy_analysis_picture_list is not None:
            print(f'accuracy_analysis_picture_list: {self.accuracy_analysis_picture_list}')
            rknn.accuracy_analysis(inputs=self.accuracy_analysis_picture_list)

        # Release
        rknn.release()

    def clean(self):
        for file_name in self.temp_files_list:
            file_path = os.path.join(self.tmp_dir, file_name)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"deleted tmp file {file_path}")
            except Exception as e:
                print(f"failed to delete {file_path}: {str(e)}")