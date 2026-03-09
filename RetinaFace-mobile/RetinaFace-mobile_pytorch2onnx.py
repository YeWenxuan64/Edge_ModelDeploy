import os
import sys
import onnx
import onnxslim

current_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录的绝对路径

onnx_model_path = 'models_convert/original/RetinaFace_mobile320.onnx'
onnx_model_path = os.path.join(current_path, onnx_model_path)

output_path = 'models_convert/onnx/RetinaFace_mobile_[1,3,320,320].onnx'
output_path = os.path.join(current_path, output_path)


# 3. 定义一个上下文管理器以安全地更改目录
class temporary_chdir:
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = None
        
    def __enter__(self):
        self.saved_path = os.getcwd() # 保存进入前的当前目录
        os.chdir(self.new_path)       # 切换到新目录
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)     # 无论代码块是否报错，都恢复原来的目录


def export(use_original_project:bool=True):
    import torch
    torch.set_grad_enabled(False)
    device = torch.device("cpu")

    if use_original_project:
        project_root = os.path.join(current_path, 'models_convert/original/Pytorch_Retinaface')
        weight_path = os.path.join(project_root, 'weights/mobilenet0.25_Final.pth')
    else:
        project_root = os.path.join(current_path, 'models_convert/original/Face-Detector-1MB-with-landmark')
        weight_path = os.path.join(project_root, 'weights/RBF_Final.pth')


    sys.path.insert(0, project_root)
    from data import cfg_mnet
    from models.retinaface import RetinaFace
    

    with temporary_chdir(project_root):
        net = RetinaFace(cfg=cfg_mnet, phase='test')

    pretrained_dict:dict = torch.load(weight_path, weights_only=True, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['state_dict']

    new_pretrained_dict = {}
    for key, value in pretrained_dict.items():
        # 如果键以'module.'开头，则去掉这个前缀
        if key.startswith('module.'):
            new_key = key.split('module.', 1)[-1]
        else:
            new_key = key
        new_pretrained_dict[new_key] = value
    pretrained_dict = new_pretrained_dict


    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    model = net.to(device)
    print('Finished loading model!')

    
    inputs = torch.randn(1, 3, 320, 320).to(device)
    torch.onnx.export(model, inputs, onnx_model_path, input_names=["input0"], output_names=["output0"], export_params=True, do_constant_folding=True)
    print('Export success')



def simplify():
    model = onnx.load_model(onnx_model_path)

    output_list = []
    for i, output in enumerate(model.graph.output):
        old_name = output.name
        new_name = f'output{i}'
        output.name = new_name
        output_list.append((old_name, new_name))

    for node in model.graph.node:
        for (old_name, new_name) in output_list:
            if node.output[0] == old_name:
                print(node.output)
                node.output[0] = new_name
    
    #验证模型有效性
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)

    print('start simplify')
    onnxslim.slim(model, output_path)

    print('Simplified')
    print(f"==> Exporting model to ONNX format at '{output_path}'")

if __name__ == '__main__':
    export()
    simplify()
