import os
import sys
import torch
import onnx
import onnxslim
from onnxslim.utils import summarize_model, print_model_info_as_table

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))


# 获取项目根目录的绝对路径 (假设脚本在 convert_to_rknn 目录下)
project_root = os.path.abspath(os.path.join(current_path, 'models_convert/original/SiamTrackers/NanoTrack')) # 上一级目录
print(project_root)

# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)


from models_convert.original.SiamTrackers.NanoTrack.nanotrack.core.config import cfg
try:
    from nanotrack.core.config import cfg
except Exception as e:
    print(e)

from models_convert.original.SiamTrackers.NanoTrack.nanotrack.utils.model_load import load_pretrain
from models_convert.original.SiamTrackers.NanoTrack.nanotrack.models.model_builder import ModelBuilder


EXPORT_FROM_PYTORCH = ['models_convert/original/NanoTrackV3_backbone_X_255_from_pytorch.onnx',
                       'models_convert/original/NanoTrackV3_backbone_T_127_from_pytorch.onnx',
                       'models_convert/original/NanoTrackV3_head_from_pytorch.onnx']
EXPORT_FROM_PYTORCH = [os.path.join(current_path, path) for path in EXPORT_FROM_PYTORCH]

EXPORT_AS_ONNX = ['models_convert/onnx/NanoTrackV3_backbone_X_255.onnx',
                  'models_convert/onnx/NanoTrackV3_backbone_T_127.onnx',
                  'models_convert/onnx/NanoTrackV3_head.onnx']
EXPORT_AS_ONNX = [os.path.join(current_path, path) for path in EXPORT_AS_ONNX]

yaml_config_path = os.path.join(current_path, 'models_convert/original/SiamTrackers/NanoTrack/models/config/configv3.yaml')
pytorch_model_path = os.path.join(current_path, 'models_convert/original/SiamTrackers/NanoTrack/models/pretrained/nanotrackv3.pth')

def export():
    cfg.merge_from_file(yaml_config_path)

    model = ModelBuilder()
    model = load_pretrain(model, pytorch_model_path)
    model.eval().to('cpu')

    backbone_net = model.backbone
    head_net = model.ban_head
    
    # backbone 图像特征提取模型
    print('convert backbone_X model to onnx')
    backbone_x_input = torch.randn([1, 3, 255, 255], device='cpu')
    torch.onnx.export(backbone_net, backbone_x_input, EXPORT_FROM_PYTORCH[0], 
                      input_names=['input'], output_names=['output'], opset_version=13, do_constant_folding=True)


    # backbone 模板特征提取模型
    print('convert backbone_T model to onnx')
    backbone_t_input = torch.randn([1, 3, 127, 127], device='cpu')
    torch.onnx.export(backbone_net, backbone_t_input, EXPORT_FROM_PYTORCH[1], 
                      input_names=['input'], output_names=['output'], opset_version=13, do_constant_folding=True)


    # head 模型
    print('convert head model to onnx')
    head_zf_input, head_xf_input = torch.randn([1, 96, 8, 8], device='cpu'), torch.randn([1, 96, 16, 16], device='cpu')
    torch.onnx.export(head_net, (head_zf_input, head_xf_input), EXPORT_FROM_PYTORCH[2], 
                      input_names=['input_z','input_x'], output_names=['output_cls','output_reg'], opset_version=13, do_constant_folding=True)


def simplify():
    print('start simplify')
    for import_model, export_model in zip(EXPORT_FROM_PYTORCH, EXPORT_AS_ONNX):
        model = onnx.load_model(import_model)
        original_info = summarize_model(model, os.path.basename(import_model))

        model = onnxslim.slim(model)

        new_model = onnx.helper.make_model(model.graph, producer_name=model.producer_name, opset_imports=[onnx.helper.make_opsetid("", 15)])
        onnx.save_model(new_model, export_model)

        slimmed_info = summarize_model(new_model, os.path.basename(export_model))
        print(f'simplify {import_model} to {export_model}')
        print_model_info_as_table([original_info, slimmed_info])


if __name__ == '__main__':
    export() 
    simplify()
