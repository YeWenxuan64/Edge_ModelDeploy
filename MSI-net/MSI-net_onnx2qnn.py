import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_qnn import OnnxToQNN

MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/MSI-net_from_tf.onnx')

QNN_MODEL = os.path.join(current_path, 'models_convert/qnn/MSI-net_i8[1,3,160,320].bin')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets_full.txt')



onnx_to_qnn = OnnxToQNN(MODEL_PATH, QNN_MODEL, DATASET_PATH)

onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])