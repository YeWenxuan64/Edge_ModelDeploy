import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_rknn import OnnxToRKNN




MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/RetinaFace_mobile_[1,3,320,320].onnx')

RKNN_MODEL = os.path.join(current_path, 'models_convert/rknn/RetinaFace_mobile_i8[1,320,320,3].rknn')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets_face.txt')
DATASET_PATH = None

TARGET_PLATFORM = 'rk3588'


onnx_to_rknn = OnnxToRKNN(MODEL_PATH, RKNN_MODEL, DATASET_PATH, TARGET_PLATFORM)

# onnx_to_rknn.do_hybrid_quantization(custom_hybrid=[['onnx::Conv_381', 'output0'],
# 													['onnx::Conv_381', 'output1'],
# 													['onnx::Conv_381', 'output2']])

onnx_to_rknn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])


