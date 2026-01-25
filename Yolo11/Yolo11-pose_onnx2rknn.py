import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_rknn import OnnxToRKNN


model_quantities = ['s', 'm']

# 模型文件路径
MODEL_PATHS = [f'models_convert/onnx/yolo11{size}-pose_[1,3,320,640].onnx' for size in model_quantities]
MODEL_PATHS = [os.path.join(current_path, path) for path in MODEL_PATHS]

# 导出路径
RKNN_MODELS = [f'models_convert/rknn/yolo11{size}-pose_i8[1,3,320,640].rknn' for size in model_quantities]
RKNN_MODELS = [os.path.join(current_path, path) for path in RKNN_MODELS]

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets.txt')

TARGET_PLATFORM = 'rk3588'



for i, (model_path, rknn_model) in enumerate(zip(MODEL_PATHS, RKNN_MODELS)):
    print("turns:", i + 1)

    onnx_to_rknn = OnnxToRKNN(model_path, rknn_model, DATASET_PATH, TARGET_PLATFORM)
    
    onnx_to_rknn.extra_optimize(flash_attantion=True)
    onnx_to_rknn.do_hybrid_quantization(custom_hybrid=[['/model.23/cv4.0/cv4.0.0/act/Mul_output_0', '/model.22/Concat_6_output_0'],
                                                      ['/model.23/cv4.1/cv4.1.0/act/Mul_output_0', '/model.22/Concat_6_output_0'],
                                                      ['/model.23/cv4.2/cv4.2.0/act/Mul_output_0', '/model.22/Concat_6_output_0']])
    
    onnx_to_rknn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])



