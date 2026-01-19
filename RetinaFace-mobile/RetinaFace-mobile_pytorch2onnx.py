import os
import onnx
import onnxslim

current_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录的绝对路径

onnx_model_path = 'models_convert/original/RetinaFace_mobile320.onnx'
onnx_model_path = os.path.join(current_path, onnx_model_path)

output_path = 'models_convert/onnx/RetinaFace_mobile_[1,3,320,320].onnx'
output_path = os.path.join(current_path, output_path)


def export():
    pass

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
