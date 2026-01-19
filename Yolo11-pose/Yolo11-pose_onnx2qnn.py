import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_qnn import OnnxToQNN


model_quantities = ['s', 'm']

# 模型文件路径
MODEL_PATHS = [f'models_convert/onnx/yolo11{size}-pose_[1,3,320,640].onnx' for size in model_quantities]
MODEL_PATHS = [os.path.join(current_path, path) for path in MODEL_PATHS]

# 导出路径
QNN_MODELS = [f'models_convert/qnn/yolo11{size}-pose_i8[1,3,320,640].bin' for size in model_quantities]
QNN_MODELS = [os.path.join(current_path, path) for path in QNN_MODELS]

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets.txt')


for i, (model_path, qnn_model) in enumerate(zip(MODEL_PATHS, QNN_MODELS)):
    print("turns:", i + 1)

    onnx_to_qnn = OnnxToQNN(model_path, qnn_model, DATASET_PATH)

    onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])