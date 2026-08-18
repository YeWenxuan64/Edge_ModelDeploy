import os
import sys
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2


current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录的绝对路径
sys.path.append(current_dir)

from utils import letterbox_image, read_dataset_txt_to_list, clean_files_or_dirs, OnnxExecutor



class ProcessDatasetByModel:
    def __init__(self, model_path:str, dataset_path:str|list[str]|None=None, output_dir_name:str='processed_by_model'):
        self.model_path = Path(model_path).resolve()
        self.output_dir_name = output_dir_name
        self.file_or_dir_to_clean = []

        if dataset_path:
            if isinstance(dataset_path, list):
                self.dataset_list_path = dataset_path
            else:
                self.dataset_list_path = read_dataset_txt_to_list(dataset_path)
        else:
            self.dataset_list_path = None

        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径
        self.output_dir = self.tmp_dir / self.output_dir_name

        self.loop_pair = None
        self.loop_inited = False
        self.replace_out_dataset_by_input = False

        # 后台写入线程池（单线程，按序写入磁盘）
        self.write_executor = ThreadPoolExecutor(max_workers=6)

    def set_ring_loop(self, loop_pair:list[int, int]):
        """
        Args:
            loop_pair: [model_input_n, model_output_n]
        """

        self.loop_pair = loop_pair
        self.output_tensor_to_loop = np.zeros((224, 224, 3), dtype=np.float32)

    def set_replace_out_dataset_by_input(self, not_normalize:bool=True):
        """
        replace the output dataset by input tensor.

        """
        self.replace_out_dataset_by_input = True
        self.replace_not_normalize = not_normalize

    def replace_output_dateset_list(self, output_dateset_list:list[np.ndarray], replace_pair_index:int):
        self.input_tensor_list_to_out = output_dateset_list[replace_pair_index]

    @staticmethod
    def _write_files(write_buffer:list[tuple[str, np.ndarray]], output_format:str):
        """实际执行写入磁盘的工作函数（在后台线程中运行）"""
        for output_path, output in write_buffer:
            if output_format == '.npy':
                np.save(output_path, output)
            elif output_format == '.raw':
                output.tofile(output_path)

        print(f"written {len(write_buffer)} {output_format} files.")

    def _flush_writes(self, write_buffer:list[tuple[str, np.ndarray]], output_format:str):
        """将缓存的 (路径, 数据) 列表提交到后台线程写入磁盘（不阻塞主流程）"""
        if not write_buffer:
            return
        # 拷贝一份，避免主线程随后 clear 影响正在执行的写入任务
        self.write_executor.submit(self._write_files, copy.deepcopy(write_buffer), output_format)

    def process(self, rgb_mean:list[list[int]]=[[0, 0, 0]], rgb_std:list[list[int]]=[[1, 1, 1]], output_order:str='chw', output_format:str='.npy', output_list:bool=False) -> str|list[list[str]]:
        if output_order not in ['chw', 'hwc', 'nchw', 'nhwc']:
            raise ValueError("output_shape must be 'chw' or 'hwc' or 'nchw' or 'nhwc'")

        if output_format not in ['.npy', '.raw']:
            raise ValueError("output_format must be '.npy' or '.raw'")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_or_dir_to_clean.append(self.output_dir)

        onnx_executor = OnnxExecutor(str(self.model_path))
        input_shapes = onnx_executor.get_input_shapes()

        output_path_pairs_list:list[list[str]] = []
        write_buffer:list[tuple[str, np.ndarray]] = []

        for i, dataset_path_pair in enumerate(self.dataset_list_path):
            input_tensor_list:list[np.ndarray] = []
            output_path_pair_list:list[str] = []

            if self.replace_out_dataset_by_input:
                self.input_tensor_list_to_out:list[np.ndarray] = []

            for j in range(len(input_shapes)):
                if not self.loop_pair or self.loop_pair[0] != j:
                    image_path = dataset_path_pair[j]
                    input_size = input_shapes[j][2:4][::-1]
                    rgb_mean = np.array(rgb_mean[j]).reshape(1, 3, 1, 1)
                    rgb_std = np.array(rgb_std[j]).reshape(1, 3, 1, 1)

                    # 读取图片
                    image = cv2.imread(image_path)
                    cv2.imshow("image", image)
                    cv2.waitKey(1)

                    if image is None:
                        print(f"无法读取图片: {image_path}")
                        continue

                    img_float = letterbox_image(image, input_size, output_format="nchw", output_dtype='float32')
                    img_norm = (img_float - rgb_mean) / rgb_std
                    tensor_ori = img_float.copy()

                else:
                    if not self.loop_inited:
                        self.loop_inited = True
                        self.output_tensor_to_loop.resize(input_shapes[self.loop_pair[0]])

                    img_norm = self.output_tensor_to_loop.copy()
                    tensor_ori = img_norm.copy()

                input_tensor_list.append(img_norm)

                if self.replace_out_dataset_by_input:
                    if self.replace_not_normalize:
                        self.input_tensor_list_to_out.append(tensor_ori)
                    else:
                        self.input_tensor_list_to_out.append(img_norm.copy())

            output_tensor_list = onnx_executor.put(input_tensor_list, input_format="nchw")


            
            for k, output_tensor in enumerate(output_tensor_list):
                if self.loop_pair and self.loop_pair[1] == k:
                    self.output_tensor_to_loop = output_tensor.copy()

                if self.replace_out_dataset_by_input:
                    output_tensor = self.input_tensor_list_to_out[k]


                if output_order == 'chw':
                    output_tensor = output_tensor.squeeze(0)
                elif output_order == 'hwc':
                    output_tensor = output_tensor.squeeze(0).transpose(1, 2, 0)
                elif output_order == 'nchw':
                    pass
                elif output_order == 'nhwc':
                    output_tensor = output_tensor.transpose(0, 2, 3, 1)

                image_name = Path(image_path).stem

                if self.replace_out_dataset_by_input:
                    output_path = str(self.output_dir / f"{image_name}_in{k}{output_format}")
                else:
                    output_path = str(self.output_dir / f"{image_name}_out{k}{output_format}")

                output_path_pair_list.append(str(output_path))
                write_buffer.append((output_path, output_tensor))

                if len(write_buffer) >= 16:
                    self._flush_writes(write_buffer, output_format)
                    write_buffer.clear()

            output_path_pairs_list.append(output_path_pair_list)

        # 刷新剩余不足 8 个的缓存
        self._flush_writes(write_buffer, output_format)

        # 等待后台线程把剩余写入任务全部落盘
        self.write_executor.shutdown(wait=True)

        cv2.destroyAllWindows()
        onnx_executor.release()

        if output_list:
            return output_path_pairs_list
        else:
            output_txt = self.tmp_dir / "datasets_processed_by_model.txt"
            self.file_or_dir_to_clean.append(output_txt)

            with open(output_txt, 'w', encoding='utf-8') as f:
                for pair_path in output_path_pairs_list:
                    pair_path_full = " ".join(pair_path)
                    pair_path_full += "\n"
                    f.write(pair_path_full)

            return str(output_txt)

    def clean(self):
        clean_files_or_dirs(self.file_or_dir_to_clean)

if __name__ == '__main__':
    model_path = "utilities/yolo26s_f16([[640,640]],[[1,300,6]]).onnx"
    model_path = "unisal_ModelDeploy/models_convert/onnx/unisal_[[1,3,160,320][1,256,5,10]].onnx"
    dataset = "datasets/datasets_short.txt"

    process_dataset = ProcessDatasetByModel(model_path, dataset)
    process_dataset.set_ring_loop([1,1])
    process_dataset.set_replace_out_dataset_by_input()
    process_dataset.process(rgb_mean=[[0, 0, 0]], rgb_std=[[1, 1, 1]], output_order='nchw', output_format='.npy')

    # npy = np.load("utilities/tmp/processed_by_model/000000000785_in1.npy")
    # print(npy.shape, npy)


# ============================================================================
# yolo_cropped_dataset_gen.py 内容已合并到此文件
# ============================================================================

import shutil
from copy import deepcopy

class OnnxExecutor:
    """占位：实际实现在 utils.py 中"""
    pass

def preprocess_image(image:np.ndarray, input_size:tuple[int, int], mean_rgb:list[int]=[0, 0, 0], std_rgb:list[int]=[1, 1, 1]) -> tuple[np.ndarray, float, int, int]:
    """预处理图像：调整大小、归一化、增加批次维度"""
    padded = letterbox_image(image, (input_size[0], input_size[1]), output_format='hwc', output_dtype='uint8', border_value=(114, 114, 114))

    h, w = image.shape[:2]
    r = min(input_size[1] / h, input_size[0] / w)
    new_h, new_w = int(h * r), int(w * r)

    # 计算填充大小 (letterbox_image 内部已做居中填充, 这里仅需坐标映射参数)
    pad_h = input_size[1] - new_h
    pad_w = input_size[0] - new_w
    top = pad_h // 2
    left = pad_w // 2

    
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
    results_num = len(results)
    if results_num:
        clipped_results_num = min(results_num, 3)
        sorted_indices = np.argsort(final_scores)[::-1][:clipped_results_num]
        results = [results[i] for i in sorted_indices]

    return results




class GenYoloCroppedDataset:
    def __init__(self, dataset_path:str, output_dir_name:str='cropped_images'):
        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
        self.tmp_dir = Path(os.path.join(current_dir, 'tmp')) # 构建tmp目录的绝对路径

        self.yolo_model_path = os.path.join(current_dir, 'yolo26s_f16([[640,640]],[[1,300,6]]).onnx')
        self.dataset_path = Path(dataset_path).resolve()
        self.output_dir_name = output_dir_name
        self.file_or_dir_to_clean = []

        self.another_args_list:list[tuple[str, str, str, str, list[int|float,], list[int|float,]]] = []

        self.human_list = [0]
        self.animal_list = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
        self.vehicle_list = [2, 3, 4, 5, 6, 7, 8, 30, 31, 33, 36, 37]

    def set_postprocess_by_another_model(self, another_model_path:str, process_target:str, output_shape:str='chw',outpur_format:str='.npy', rgb_mean:list[int]=[0, 0, 0], rgb_std:list[int]=[1, 1, 1]):
        """
        Args:
            another_model_path_and_target_list (list[tuple[str, str]]): [(another_model_path, process_target), ...]
                - process_target (str): 'output' or 'input'
            output_shape (str): 'chw' or 'hwc' or 'nchw' or 'nhwc'
            outpur_format (str): '.npy' or '.raw'
        """
        if process_target not in ['output', 'input']:
            raise ValueError("process_target must be 'output' or 'input'")

        if output_shape not in ['chw', 'hwc', 'nchw', 'nhwc']:
            raise ValueError("output_shape must be 'chw' or 'hwc' or 'nchw' or 'nhwc'")

        if outpur_format not in ['.npy', '.raw']:
            raise ValueError("outpur_format must be '.npy' or '.raw'")

        another_model_path = Path(another_model_path).resolve()
        self.another_args_list.append((str(another_model_path), process_target, output_shape, outpur_format, rgb_mean, rgb_std))
        
    @staticmethod
    def postprocess_by_another_model(another_model_path:str, image_path_list:list[str], output_shape, output_format, mean_rgb, std_rgb) -> list[str]:
        onnx_executor = OnnxExecutor(another_model_path)
        input_sizes = onnx_executor.get_input_shapes()[0][2:4][::-1]

        output_path_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            cv2.imshow('image', image)
            cv2.waitKey(1)
            input_tensor, scale, x_offset, y_offset = preprocess_image(image, input_sizes, mean_rgb=mean_rgb, std_rgb=std_rgb)
            
            outputs = onnx_executor.put([input_tensor], input_format="nchw")
            output:np.ndarray = outputs[0] # nchw

            if output_shape == 'chw':
                output = output.squeeze()
            elif output_shape == 'hwc':
                output = output.squeeze().transpose(1, 2, 0)
            elif output_shape == 'nchw':
                pass
            elif output_shape == 'nhwc':
                output = output.transpose(0, 2, 3, 1)

            output_path = str(Path(image_path).with_suffix(output_format))

            if output_format == '.npy':
                np.save(output_path, output)

            elif output_format == '.raw':
                output.tofile(output_path)

            output_path_list.append(output_path)

        onnx_executor.release()
        return output_path_list

    def prepare_work_dir(self) -> tuple[list[str], str]:
        self.tmp_dir.mkdir(parents=True, exist_ok=True) # 创建tmp目录
        # 创建输出目录

        output_dir = os.path.join(self.tmp_dir, self.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

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

        input_images_dir = os.path.join(self.tmp_dir, 'input_images')
        os.makedirs(input_images_dir, exist_ok=True)

        # 遍历图片输入列表并复制文件
        count_copied_files: dict[str, int] = {}

        for i, src_path in enumerate(full_img_path_list):
            parent_dir = os.path.dirname(src_path)
            file_name = os.path.basename(src_path) # 获取文件名（例如 'img1.jpg'）

            dst_path = os.path.join(input_images_dir, file_name) # 拼接目标路径
            full_img_path_list[i] = dst_path
            
            try:
                shutil.copy2(src_path, dst_path) # 执行复制操作
                count_copied_files[parent_dir] = count_copied_files.get(parent_dir, 0) + 1

            except Exception as e:
                print(f"failed to copy {src_path}: {e}")

        # 按源目录打印汇总信息
        for srcdir, n in count_copied_files.items():
            plural = "s" if n > 1 else ""
            print(f"copied {n} file{plural} from {srcdir} to {input_images_dir}")

        self.file_or_dir_to_clean.extend([input_images_dir, output_dir])
        return full_img_path_list, output_dir

    def crop_and_save(self, image: np.ndarray, boxes: list[list[int, int, int, int, int, float]],
                        scale: float, x_offset: int, y_offset: int, output_dir: str, image_name: str) -> list[str]:
        
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

            output_path = os.path.join(output_dir, f"{image_name}_crop_{i}.jpg")
            cv2.imwrite(output_path, cropped)
                
            saved_paths.append(os.path.abspath(output_path))
        
        # 显示带有边界框的图像
        cv2.imshow('Detection Results', display_image)
        cv2.waitKey(1)
        
        return saved_paths

    def generate(self, swap_image_pair:bool=False, save_original_path_pair:bool=False) -> str|tuple[str, str]:
        """
        Args:
            swap_image_pair (bool, optional): 是否交换得到的输入和输出图片对. Defaults to False.
            - True: cropped_path full_img_path
            - False: full_img_path cropped_path

            save_original_path_pair (bool, optional): 是否保存原始图片路径对, 在 set_postprocess_by_another_model 后可用. Defaults to False.

        Returns:
            str: 生成的数据集路径文本文件的路径
        """

        full_img_path_list, output_dir = self.prepare_work_dir()

        # 初始化ONNX运行时
        onnx_executor = OnnxExecutor(self.yolo_model_path)
        input_sizes = onnx_executor.get_input_shapes()[0][2:4][::-1]
        all_cropped_paths:list[list[str, str]] = []
        for image_path in full_img_path_list:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            # 预处理
            input_tensor, scale, x_offset, y_offset = preprocess_image(image, input_size=input_sizes, std_rgb=[255, 255, 255])
            
            # 推理
            output = onnx_executor.put([input_tensor], input_format="nchw")[0]
            
            # 处理预测结果
            results = process_predictions(output)

            if results:
                # 裁剪并保存
                image_name = Path(image_path).stem

                cropped_paths = self.crop_and_save(image, results, scale, x_offset, y_offset, output_dir, image_name)

                if cropped_paths:
                    for cropped_path in cropped_paths:
                        all_cropped_paths.append([image_path, cropped_path])

        cv2.destroyWindow("Detection Results")
        onnx_executor.release()

        original_all_cropped_paths = deepcopy(all_cropped_paths)
        new_all_cropped_paths = deepcopy(all_cropped_paths)
        if self.another_args_list:
            for i, (another_model_path, process_target, output_shape, output_format, mean_rgb, std_rgb) in enumerate(self.another_args_list):
                print(f"Processing by another model: {another_model_path} ...")

                process_path_list:list[str] = []
                if process_target == 'input': # 匹配裁切前的图片
                    index = 0
                elif process_target == 'output': # 匹配裁切后的图片
                    index = 1

                for pair_path in all_cropped_paths:
                    process_path_list.append(pair_path[index])

                output_path_list = self.postprocess_by_another_model(another_model_path, process_path_list, output_shape, output_format, mean_rgb, std_rgb)

                for j, pair_path in enumerate(new_all_cropped_paths):
                    pair_path[index] = output_path_list[j] # 替换为处理后的文件的路径

        cv2.destroyAllWindows()
    
        # 保存裁剪图片的路径列表
        output_txt = str(self.tmp_dir / str(self.output_dir_name + '_list.txt'))
        self.file_or_dir_to_clean.append(output_txt)

        with open(output_txt, 'w', encoding='utf-8') as f:
            for pair_path in new_all_cropped_paths:
                if swap_image_pair:
                    pair_path = pair_path[::-1]

                pair_path_full = f"{pair_path[0]} {pair_path[1]}\n"
                f.write(pair_path_full)


        if save_original_path_pair:
            original_output_txt = str(self.tmp_dir / str(self.output_dir_name + '_original_list.txt'))
            self.file_or_dir_to_clean.append(original_output_txt)

            with open(original_output_txt, 'w', encoding='utf-8') as f:
                for pair_path in original_all_cropped_paths:
                    if swap_image_pair:
                        pair_path = pair_path[::-1]

                    pair_path_full = f"{pair_path[0]} {pair_path[1]}\n"
                    f.write(pair_path_full)

            return output_txt, original_output_txt

        else:
            return output_txt


    def clean(self):
        clean_files_or_dirs(self.file_or_dir_to_clean)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
    parent_dir = os.path.dirname(current_dir)

    # 配置参数
    dataset_path = os.path.join(parent_dir, 'datasets/datasets_face.txt')  # 输入图片索引文本

    # 另一个AI模型路径
    another_model1_path = 'nanotrack_v3_ModelDeploy/models_convert/onnx/NanoTrackV3_backbone_X_255.onnx'
    another_model2_path = 'nanotrack_v3_ModelDeploy/models_convert/onnx/NanoTrackV3_backbone_T_127.onnx'

    # 创建对象并生成数据集
    dataset_generator = GenYoloCroppedDataset(dataset_path, 'cropped_images2')
    dataset_generator.set_postprocess_by_another_model(another_model1_path, process_target="input", output_shape="nchw", outpur_format='.npy')
    dataset_generator.set_postprocess_by_another_model(another_model2_path, process_target="output", output_shape="nchw", outpur_format='.npy')

    cropped_list_path = dataset_generator.generate()

    print(f"裁剪后的图片路径列表已保存到: {cropped_list_path}")

    #dataset_generator.clean()