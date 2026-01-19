import numpy as np  # 导入NumPy库，用于数值计算
import tensorflow as tf  # 导入TensorFlow库，用于深度学习模型
import cv2  # 导入OpenCV库，用于图像处理
import time
import os

TF_ENABLE_ONEDNN_OPTS=0  # 禁用OneDNN优化，以提高兼容性

def preprocess_input(input_image):
    original_shape = input_image.shape

    target_shape = (160, 320)

    resize_ratio = max(target_shape) / max(original_shape)

    resize_image = cv2.resize(input_image, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    resize_shape = resize_image.shape

    # 计算需要在垂直和水平方向上填充的像素数
    vertical_padding = target_shape[0] - resize_shape[0]
    horizontal_padding = target_shape[1] - resize_shape[1]

    # 计算填充的上下和左右像素数
    vertical_padding_1 = vertical_padding // 2
    vertical_padding_2 = vertical_padding - vertical_padding_1

    horizontal_padding_1 = horizontal_padding // 2
    horizontal_padding_2 = horizontal_padding - horizontal_padding_1

    if vertical_padding_1 < 0:
        resize_image = resize_image[-vertical_padding_1:vertical_padding_1, :, :]
        vertical_padding_1, vertical_padding_2 = 0, 0
    if horizontal_padding_1 < 0:
        resize_image = resize_image[:, -horizontal_padding_1:horizontal_padding_1, :]
        horizontal_padding_1, horizontal_padding_2 = 0, 0

    paded_image = cv2.copyMakeBorder(resize_image, vertical_padding_1, vertical_padding_2, horizontal_padding_1, horizontal_padding_2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    cv2.imshow("Padded Image", cv2.resize(paded_image,original_shape[:2:][::-1],interpolation=cv2.INTER_LINEAR))

    input_tensor = np.expand_dims(paded_image, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    return (input_tensor, [vertical_padding_1, vertical_padding_2], [horizontal_padding_1, horizontal_padding_2],)

def postprocess_output(output_tensor, vertical_padding, horizontal_padding, original_shape):
    output_size = output_tensor.shape
    # 去除填充部分，恢复原始形状
    horizontal_slice = [vertical_padding[0], output_size[1] - vertical_padding[1]]
    vertical_slice = [horizontal_padding[0], output_size[2] - horizontal_padding[1]]
    output_tensor = output_tensor[:, horizontal_slice[0]:horizontal_slice[1], vertical_slice[0]:vertical_slice[1], :]

    # 将输出张量转换为NumPy数组并去除多余的维度
    output_array = np.squeeze(output_tensor.numpy())

    output_array = cv2.resize(output_array, original_shape[::-1])

    output_array = cv2.normalize(output_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    output_array = cv2.applyColorMap(output_array, cv2.COLORMAP_JET)

    return output_array

def main():
    # 加载保存的TensorFlow模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_dir = os.path.join(parent_dir, "/models_convert/original/save_model")
    model = tf.saved_model.load(model_dir) #"./save_model"

    # 获取模型的默认服务签名
    model = model.signatures["serving_default"]

    alpha = 0.4

    last_print_time = time.time()

    # 打开摄像头
    cap = cv2.VideoCapture("./loco640.mp4")

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

        original_shape = frame.shape[:2]

        input_tensor, vertical_padding, horizontal_padding = preprocess_input(frame)

        start_time = time.time()
        output_tensor = model(input_tensor)['layer_from_saved_model']


        saliency_map = postprocess_output(output_tensor, vertical_padding, horizontal_padding, original_shape)


        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        blended_image = cv2.addWeighted(frame, 1 - alpha, saliency_map, alpha, 0)

        end_time = time.time()

        # 使用OpenCV显示图片
        #cv2.imshow("Input Frame", frame)
        cv2.imshow("Saliency Map", blended_image)

        persent_time = time.time()
        if persent_time - last_print_time > 1:
            frame_rate = 1 / (end_time - start_time + 0.00000001)
            print(f"Frame Rate: {frame_rate:.4f} FPS")
            last_print_time = persent_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
