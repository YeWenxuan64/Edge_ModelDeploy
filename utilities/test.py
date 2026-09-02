import sys
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import read_txt_line


def test_aimet_to_qnn(target_platform: str = 'qcs6490',
                      num_calibration_samples: int | None = None) -> str:
    """
    接入测试：RetinaFace_mobile 使用 AIMET 2.x 量化 -> QAIRT 转 QNN（DLC + context binary）。

    流程（set_use_aimet 启用后由 OnnxToQNN.convert 自动执行）：
        ONNX（已烘焙归一化）-> AIMET PTQ 量化 -> QDQ ONNX + encodings
        -> qairt-converter 直接转成量化 DLC（跳过 qairt-quantizer）-> context binary

    注意：需要 QAIRT SDK（utilities/qairt）且脚本在真实终端运行
    （convert 内部会 source envsetup.sh 并调用 qairt-converter / qnn-context-binary-generator）。

    Args:
        target_platform (str): 'qcs6490' / 'qcs8550' / 'qcs9075'。默认 'qcs6490'。
        num_calibration_samples (int | None): AIMET 校准样本数上限；None 使用全部。

    Returns:
        QNN context binary 路径（.bin）。
    """
    project_dir = Path(__file__).resolve().parent.parent
    model_path = (project_dir / 'retinaface_mobile_ModelDeploy/models_convert/onnx'
                  / 'RetinaFace_mobile_[1,3,320,320].onnx')

    # model_path = current_dir / "tmp/RetinaFace_short.onnx"
    dataset_path = project_dir / 'datasets/datasets_face.txt'
    qnn_model_path = current_dir / "tmp/RetinaFace.bin"

    from onnx_to_qnn import OnnxToQNN  # 延迟导入，避免独立使用测试强依赖 QAIRT
    mean_rgb = [[123.675, 116.28, 103.53]]
    std_rgb = [[1, 1, 1]]

    converter = OnnxToQNN(str(model_path), str(qnn_model_path), str(dataset_path),
                          target_platform=target_platform)
    converter.set_quantization_method(bitwidth="w8a8", param_quant_method='sqnr', act_quant_method="entropy")
    # 启用 AIMET 2.x 量化路径（替代 qairt-quantizer；
    # 全局位宽复用上方 set_quantization_method 设置的结果）
    # converter.set_use_aimet(quant_method='tf_enhanced', act_quant_schema="unsignedsymmetric") # sequential_mse tf_enhanced

    # converter.do_hybrid_quantization([["/fpn/output3/output3.2/LeakyRelu_output_0", '/ssh3/Concat_output_0'],
    #                                   ["/fpn/output2/output2.2/LeakyRelu_output_0", '/ssh2/Concat_output_0'],
    #                                   ["/fpn/output1/output1.2/LeakyRelu_output_0", '/ssh1/Concat_output_0']],
    #                                   bitwidth="w16a16")


    converter.set_do_accuracy_analysis([str(project_dir / 'datasets/face.jpg')]) # [str(project_dir / 'datasets/face.jpg')] read_txt_line(dataset_path)
    converter.convert(mean_rgb=mean_rgb, std_rgb=std_rgb, set_input_order='nhwc')
    # converter.clean()
    print(f"\nDone. QNN model: {qnn_model_path}")
    return str(qnn_model_path)

test_aimet_to_qnn()