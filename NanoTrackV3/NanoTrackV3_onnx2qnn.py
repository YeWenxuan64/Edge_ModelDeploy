import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_qnn import OnnxToQNN
from utilities.yolo_det_dataset_gen import GenYoloDetedDataset

# 模型文件路径
MODEL_PATHS = ['models_convert/onnx/NanoTrackV3_backbone_X_255.onnx', 
              'models_convert/onnx/NanoTrackV3_backbone_T_127.onnx', 
              'models_convert/onnx/NanoTrackV3_head.onnx']
MODEL_PATHS = [os.path.join(current_path, path) for path in MODEL_PATHS]

# 导出路径
QNN_MODELS = ['models_convert/qnn/NanoTrackV3_backbone_X_i8[1,255,255,3].bin', 
              'models_convert/qnn/NanoTrackV3_backbone_T_i8[1,127,127,3].bin', 
              'models_convert/qnn/NanoTrackV3_head_i8[[1,96,8,8][1,96,16,16]].bin']
QNN_MODELS = [os.path.join(current_path, path) for path in QNN_MODELS]

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets.txt')
DATASET_2 = os.path.join(os.path.dirname(current_path), 'datasets/datasets_face.txt')
DATASET_2_PATH = []


last_index = len(MODEL_PATHS) - 1




# 循环处理每个模型
for i, (model_path, qnn_model) in enumerate(zip(MODEL_PATHS, QNN_MODELS)):
    print("turns:", i + 1)

    if i != last_index:
        onnx_to_qnn = OnnxToQNN(model_path, qnn_model, DATASET_PATH)
        onnx_to_qnn.set_quantization_method(param_quant_method='entropy', act_quant_method='entropy')

    else:
        # 创建对象并生成数据集
        for j in range(2):
            dataset_generator = GenYoloDetedDataset(DATASET_2, f'cropped_images_{j}')
            dataset_generator.set_postprocess_by_another_ai(MODEL_PATHS[j], "nhwc", '.raw')
            cropped_list_path = dataset_generator.gerenate()
            DATASET_2_PATH.append(cropped_list_path)

        # 读取两个文件的内容
        with open(DATASET_2_PATH[1], 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        
        with open(DATASET_2_PATH[0], 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()

        tmp_dataset_path = os.path.join(os.path.dirname(DATASET_2_PATH[0]), 'cropped_images.txt')

        # 合并内容并写入新文件
        with open(tmp_dataset_path, 'w', encoding='utf-8') as out_file:
            for k in range(min(len(lines1), len(lines2))):
                line1 = lines1[k]
                line2 = lines2[k]

                # 去除每行末尾的换行符，添加空格，然后合并
                merged_line = line1.rstrip() + ' ' + line2.rstrip() + '\n'
                out_file.write(merged_line)


        onnx_to_qnn = OnnxToQNN(model_path, qnn_model, tmp_dataset_path)
        onnx_to_qnn.set_quantization_method(param_quantization_method='entropy', act_quantization_method='entropy')
        onnx_to_qnn.use_custom_alibration_data(tmp_dataset_path)


    if i != last_index:
        onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
    else:
        onnx_to_qnn.convert(mean_rgb=[[0]*96,[0]*96], std_rgb=[[1]*96,[1]*96])

