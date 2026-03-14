import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_qnn import OnnxToQNN


model_quantities = ['s', 'm']

# 模型文件路径
MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/yolo11s-pose_[1,3,320,640].onnx')

# 导出路径
QNN_MODEL = os.path.join(current_path, 'models_convert/qnn/yolo11s-pose_i8[1,320,640,3].bin')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets.txt')


if __name__ == '__main__':
    onnx_to_qnn = OnnxToQNN(MODEL_PATH, QNN_MODEL, DATASET_PATH)

    onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])