import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_rknn import OnnxToRKNN


MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/MSI-net_[1,3,160,320].onnx')

RKNN_MODEL = os.path.join(current_path, 'models_convert/rknn/MSI-net_i8[1,3,160,320].rknn')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets_full.txt')

TARGET_PLATFORM = 'rk3588'


onnx_to_rknn = OnnxToRKNN(MODEL_PATH, RKNN_MODEL, DATASET_PATH, TARGET_PLATFORM)
onnx_to_rknn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])