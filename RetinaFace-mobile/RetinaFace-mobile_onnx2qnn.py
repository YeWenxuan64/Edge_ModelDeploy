import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)
parent_path = os.path.dirname(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_qnn import OnnxToQNN


MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/RetinaFace_mobile_[1,3,320,320].onnx')

QNN_MODEL = os.path.join(current_path, 'models_convert/qnn/RetinaFace_mobile_i8[1,320,320,3].bin')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets_face.txt')
accuracy_analysis_dataset = os.path.join(parent_path, 'datasets/bus.jpg')


onnx_to_qnn = OnnxToQNN(MODEL_PATH, QNN_MODEL, DATASET_PATH)
onnx_to_qnn.set_quantization_method(param_quant_method='entropy', act_quant_method='entropy')
#onnx_to_qnn.set_do_accuracy_analysis(accuracy_analysis_dataset)
onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
# onnx_to_qnn.clean()