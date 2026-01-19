import os
import sys
import torch
import onnxslim

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))


# 获取项目根目录的绝对路径 (假设脚本在 convert_to_rknn 目录下)
project_root = os.path.abspath(os.path.join(current_path, 'models_convert/original/SiamTrackers/NanoTrack')) # 上一级目录
print(project_root)

# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)


from models_convert.original.SiamTrackers.NanoTrack.nanotrack.core.config import cfg
from nanotrack.core.config import cfg

from models_convert.original.SiamTrackers.NanoTrack.nanotrack.utils.model_load import load_pretrain
from models_convert.original.SiamTrackers.NanoTrack.nanotrack.models.model_builder import ModelBuilder


EXPORT_FROM_PYTORCH = ['models_convert/original/NanoTrackV3_backbone_X_255_from_pytorch.onnx',
                       'models_convert/original/NanoTrackV3_backbone_T_127_from_pytorch.onnx',
                       'models_convert/original/NanoTrackV3_head_from_pytorch.onnx']
EXPORT_FROM_PYTORCH = [os.path.join(current_path, path) for path in EXPORT_FROM_PYTORCH]

EXPORT_AS_ONNX = ['./models_convert/onnx/NanoTrackV3_backbone_X_255.onnx',
                  './models_convert/onnx/NanoTrackV3_backbone_T_127.onnx',
                  './models_convert/onnx/NanoTrackV3_head.onnx']
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
    backbone_x = torch.randn([1, 3, 255, 255], device='cpu')
    torch.onnx.export(backbone_net, backbone_x, EXPORT_FROM_PYTORCH[0], 
                      input_names=['input'], output_names=['output'], verbose=True, opset_version=11, do_constant_folding=True)


    # backbone 模板特征提取模型
    print('convert backbone_T model to onnx')
    backbone_T = torch.randn([1, 3, 127, 127], device='cpu')
    torch.onnx.export(backbone_net, backbone_T, EXPORT_FROM_PYTORCH[1], 
                      input_names=['input'], output_names=['output'], verbose=True, opset_version=11, do_constant_folding=True)


    # head 模型
    print('convert head model to onnx')
    head_zf, head_xf = torch.randn([1, 96, 8, 8], device='cpu'), torch.randn([1, 96, 16, 16], device='cpu')
    torch.onnx.export(head_net,(head_zf,head_xf), EXPORT_FROM_PYTORCH[2], 
                      input_names=['input1','input2'], output_names=['output1','output2'],verbose=True, opset_version=11, do_constant_folding=True)


def simplify():
    print('start simplify')
    for import_model, export_model in zip(EXPORT_FROM_PYTORCH, EXPORT_AS_ONNX):
        onnxslim.slim(import_model, export_model)


if __name__ == '__main__':
    export() 
    simplify()
