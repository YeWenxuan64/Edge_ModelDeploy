import os
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

def resize_image(image:np.ndarray, input_size:tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    # 调整图像大小并保持宽高比
    h, w = image.shape[:2]
    r = min(input_size[1] / h, input_size[0] / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 计算填充大小
    pad_h = input_size[1] - new_h
    pad_w = input_size[0] - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 使用copyMakeBorder填充图像
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 灰色填充
    return padded, r, left, top

def preprocess_image(image:np.ndarray, input_size:tuple[int, int], mean_rgb:list[int]=[0, 0, 0], std_rgb:list[int]=[1, 1, 1]) -> tuple[np.ndarray, float, int, int]:
    """预处理图像：调整大小、归一化、增加批次维度"""
    padded, r, left, top = resize_image(image, input_size)
    
    # 归一化并调整维度顺序
    mean_rgb = np.array(mean_rgb, dtype=np.float32).reshape((1, 1, 3))
    std_rgb = np.array(std_rgb, dtype=np.float32).reshape((1, 1, 3))
    normalized = (padded.astype(np.float32) - mean_rgb) / std_rgb

    normalized = normalized.transpose(2, 0, 1)
    normalized = np.expand_dims(normalized, axis=0)
    
    return normalized, r, left, top

def process_predictions(output: np.ndarray) -> list[list[int, int, int, int, int, float]]:
    """处理模型输出，返回置信度最高的3个检测结果"""

    conf_threshold:float=0.25

    # 输出形状为 (1, 300, 6)
    output = output.squeeze()  # 移除批次维度，形状变为 (300, 6)

    # 分离边界框坐标、类别ID和分数
    boxes = output[:, :4]  # (300, 4)
    scores = output[:, 4]  # (300,)
    class_ids = output[:, 5]  # (300,)
    
    # 应用置信度阈值过滤
    mask = scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    if len(filtered_boxes) == 0:
        return []

    idxs = np.argsort(filtered_scores, axis=0)[::-1]  # 按置信度降序排序
    filtered_boxes = filtered_boxes[idxs]
    filtered_scores = filtered_scores[idxs]
    filtered_class_ids = filtered_class_ids[idxs]

    results = []
    final_scores = []

    for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
        x1, y1, x2, y2 = box
        
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 计算综合评分
        # 1. 中心距离得分（距离中心越近得分越高）
        center_distance = np.sqrt((cx - 320)**2 + (cy - 320)**2)
        max_distance = np.sqrt(320**2 + 320**2)
        center_score = 1 - (center_distance / max_distance)
        
        # 2. 面积得分（面积越大得分越高）
        area = w * h
        max_area = 640 * 640  # 假设最大可能面积
        area_score = area / max_area
        
        # 3. 综合得分（加权平均）
        final_score = 0.4 * score + 0.3 * center_score + 0.3 * area_score
        
        if area > 128*128:
            results.append([int(x1), int(y1), int(x2), int(y2), int(class_id), float(score)])
            final_scores.append(final_score)


    # 根据综合得分排序，取前3个
    if len(results) > 0:
        sorted_indices = np.argsort(final_scores)[::-1][:3]
        results = [results[i] for i in sorted_indices]

    return results




class GenYoloDetedDataset:
    def __init__(self, dataset_path:str, output_dir_name:str='cropped_images'):
        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径

        self.yolo_model_path = Path(os.path.join(current_dir, 'yolo26s_f32([[640,640]],[[1,300,6]]).onnx')).resolve()
        self.dataset_path = Path(dataset_path).resolve()
        self.output_dir_name = output_dir_name

        self.another_ai_onnx = None

        self.human_list = [0]
        self.animal_list = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
        self.vehicle_list = [2, 3, 4, 5, 6, 7, 8, 30, 31, 33, 36, 37]

    def set_postprocess_by_another_ai(self, another_ai_onnx_path:str, output_shape:str='chw',outpur_format:str='.npy'):
        another_ai_onnx_path = Path(another_ai_onnx_path).resolve()
        
        if output_shape not in ['chw', 'hwc', 'nchw', 'nhwc']:
            raise ValueError("output_shape must be 'chw' or 'hwc' or 'nchw' or 'nhwc'")

        if outpur_format not in ['.npy', '.raw']:
            raise ValueError("outpur_format must be '.npy' or '.raw'")
        
        model = onnx.load(another_ai_onnx_path)
        self.another_ai_input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        del model

        self.output_shape = output_shape
        self.output_format = outpur_format

        self.another_ai_onnx = ort.InferenceSession(another_ai_onnx_path)
        self.anorher_ai_input_name = self.another_ai_onnx.get_inputs()[0].name


    def postprocess_by_another_ai(self, image:np.ndarray) -> np.ndarray:
        input_h, input_w = self.another_ai_input_shape[2], self.another_ai_input_shape[3]
        input_tensor, scale, x_offset, y_offset = preprocess_image(image, (input_w, input_h), std_rgb=[1, 1, 1])

        input_name = self.anorher_ai_input_name
        outputs = self.another_ai_onnx.run(None, {input_name: input_tensor})

        output:np.ndarray = outputs[0] # nchw

        if self.output_shape == 'chw':
            output = output.squeeze()
        elif self.output_shape == 'hwc':
            output = output.squeeze().transpose(1, 2, 0)
        elif self.output_shape == 'nchw':
            pass
        elif self.output_shape == 'nhwc':
            output = output.transpose(0, 2, 3, 1)

        return output

        
    def gerenate(self) -> Path|str:
        self.tmp_dir.mkdir(parents=True, exist_ok=True) # 创建tmp目录
        # 创建输出目录

        output_dir = self.tmp_dir / self.output_dir_name
        os.makedirs(output_dir, exist_ok=True)

        # 初始化ONNX运行时
        session = ort.InferenceSession(self.yolo_model_path)
        input_name = session.get_inputs()[0].name

        # 读取图片路径列表
        image_path_set = set()
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    image_path_set.add(str(Path(line_s)))

        # 构建完整图片路径
        dataset_dir = str(self.dataset_path.parent)
        full_img_path_list = [os.path.join(dataset_dir, img_path) for img_path in image_path_set]

        all_cropped_paths = []
        for image_path in full_img_path_list:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            # 预处理
            input_tensor, scale, x_offset, y_offset = preprocess_image(image, input_size=(640, 640), std_rgb=[255, 255, 255])
            
            # 推理
            output = session.run(None, {input_name: input_tensor})[0]
            
            # 处理预测结果
            results = process_predictions(output)

            if results:
                # 裁剪并保存
                image_name = Path(image_path).stem

                cropped_paths = self.crop_and_save(image, results, scale, x_offset, y_offset, output_dir, image_name)
            all_cropped_paths.extend(cropped_paths)
        
        # 保存裁剪图片的路径列表
        output_txt = self.tmp_dir / str(self.output_dir_name + '_list.txt')
        with open(output_txt, 'w', encoding='utf-8') as f:
            for path in all_cropped_paths:
                f.write(path + '\n')

        cv2.destroyAllWindows()
        del session

        return output_txt

    def crop_and_save(self, image: np.ndarray, boxes: list[list[int, int, int, int, int, float]],
                    scale: float,
                    x_offset: int,
                    y_offset: int,
                    output_dir: str,
                    image_name: str) -> list[str]:
        """裁剪并保存检测结果"""
        saved_paths = []
        display_image = image.copy()  # 创建显示用的图像副本
        
        for i, (x1, y1, x2, y2, class_id, conf) in enumerate(boxes):
            # 将坐标转换回原始图像尺寸
            x1 = int((x1 - x_offset) / scale)
            y1 = int((y1 - y_offset) / scale)
            x2 = int((x2 - x_offset) / scale)
            y2 = int((y2 - y_offset) / scale)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            x2 = max(0, min(x2, image.shape[1]))
            y2 = max(0, min(y2, image.shape[0]))
            
            id = int(class_id)

            if id in self.human_list:
                color = (255, 0, 0)
            elif id == 1 or id in self.animal_list:
                color = (0, 255, 0)
            elif id == 2 or id in self.vehicle_list:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            # 在显示图像上绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            # 在边界框上方显示置信度
            cv2.putText(display_image, f'{conf:.2f}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            # 裁剪并保存
            cropped = image[y1:y2, x1:x2]

            cv2.imshow("Cropped Image", cropped)
            cv2.waitKey(1)

            if self.another_ai_onnx is None:
                output_path = os.path.join(output_dir, f"{image_name}_crop_{i}.jpg")
                cv2.imwrite(output_path, cropped)
                
            else:
                output_path = os.path.join(output_dir, f"{image_name}_crop_{i}{self.output_format}")
                cropped = self.postprocess_by_another_ai(cropped)
                if self.output_format == '.npy':
                    np.save(output_path, cropped)

                elif self.output_format == '.raw':
                    cropped.tofile(output_path)
                
            saved_paths.append(os.path.abspath(output_path))
        
        # 显示带有边界框的图像
        cv2.imshow('Detection Results', display_image)
        cv2.waitKey(1)
        
        return saved_paths

def main():
    # 配置参数
    dataset_path = './datasets/datasets_face.txt'  # 输入图片索引文本

    # 创建对象并生成数据集
    dataset_generator = GenYoloDetedDataset(dataset_path, 'cropped_images2')
    #dataset_generator.set_postprocess_by_another_ai('./NanoTrackV3/models_convert/onnx/NanoTrackV3_backbone_X_255.onnx', "nchw", '.npy')

    cropped_list_path = dataset_generator.gerenate()

    print(f"裁剪后的图片路径列表已保存到: {cropped_list_path}")



if __name__ == "__main__":
    main()
