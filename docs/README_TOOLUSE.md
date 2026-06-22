# 工具链使用指南

## 0. 安装依赖与准备量化校准数据集
已在[README.md](README.md)中介绍

#### 使用 `collect_image_paths` 生成数据集索引文件

下载图片后，可以用 `utilities/utils.py` 中的 `collect_image_paths()` 函数自动生成路径索引文件：

```python
from utilities.utils import collect_image_paths

# 从指定目录收集图片路径，生成 txt 索引文件
dataset_txt = collect_image_paths(
    dir_paths=['/path/to/coco/val2017'],  # 图片目录列表
    max_count=200                          # 最多取 200 张
)

print(f"数据集索引已生成: {dataset_txt}")
# 输出: utilities/tmp/combind_image_paths.txt
```

该函数会扫描指定目录下所有常见格式的图片（jpg/png/bmp/webp/tiff 等），将绝对路径逐行写入 `utilities/tmp/combind_image_paths.txt`，然后返回该文件的绝对路径。


## 1. 创建转换工程
模仿示例的子模块，新建文件夹转换工程文件夹，并按照以特定结构放置文件和填写转换脚本
```
Edge_ModelDeploy/
├── utilities/
│   └── ...
├── datasets/                       
│   └── ...                         
├── Your_ModelDeploy/                         # 新建该模型的转换工程文件夹
│   ├── models_convert/                
│   │   ├── original/                         # 原始模型目录
│   │   ├── onnx/              
│   │   └── ...
│   ├── Your_Model_originalframework2onnx.py  # PyTorch → ONNX 转换脚本
│   ├── Your_Model_onnx2rknn.py               # ONNX → RKNN 转换脚本（如果需要）
│   └── Your_Model_onnx2qnn.py                # ONNX → QNN 转换脚本（如果需要）
└── ...
```

## 2. 编写从原始框架模型到 ONNX 模型的转换脚本
模仿示例的子模块，编写导出脚本`Your_Model_originalframework2onnx.py`<br>
>从原始框架导出的模型文件放置在`models_convert/original/`目录下<br>
>优化后的 ONNX 模型文件放置在`models_convert/onnx/`目录下<br>

```python
from pathlib import Path
import onnx
import onnxslim
from onnxslim.utils import summarize_model, print_model_info_as_table

current_dir = Path(__file__).parent.resolve()
project_dir = current_dir / 'models_convert/original/your_model_train_project'

yourmodel_output_path = str(current_dir / 'models_convert/original/yourmodel_from_pytorch.onnx')
onnx_model_output_path = str(current_dir / 'models_convert/onnx/yourmodel.onnx')

def export():
    pass

def simplify():
    pass

if __name__ == '__main__':
    export()
    simplify()
```

## 3.1 `OnnxToRKNN` — ONNX 转 Rockchip RKNN
模仿示例的子模块，编写转换脚本``Your_Model_onnx2rknn.py``<br>
>之前优化后的 ONNX 模型文件放置在`models_convert/onnx/`目录下<br>
>转换后的 RKNN 模型文件放置在`models_convert/rknn/`目录下<br>

```python
from pathlib import Path
from utilities.onnx_to_rknn import OnnxToRKNN

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent

# 模型文件路径
MODEL_PATH = str(current_dir / 'models_convert/onnx/yourmodel.onnx')

# 导出路径
RKNN_MODEL = str(current_dir / 'models_convert/rknn/yourmodel_i8.rknn')

# 量化校准数据集路径（可选）
DATASET = str(parent_dir / 'datasets/datasets.txt')

TARGET_PLATFORM = 'rk3588'


converter = OnnxToRKNN(
    model_path=MODEL_PATH,
    rknn_model_path=RKNN_MODEL,
    dataset_path=DATASET,
    target_platform=TARGET_PLATFORM        # 'rk3588' / 'rk3576' / 'rk3566'
)

# 可选：高级优化配置
converter.extra_optimize(
    quantized_algorithm='kl_divergence',   # 'normal' / 'kl_divergence' / 'mmse'
    flash_attantion=True,
    compress_weight=False,
    model_pruning=False
)

# 可选：混合量化（部分子图 FP16，其余 INT8）
# converter.do_hybrid_quantization(
#     custom_hybrid=[['input_node', 'output_node']]
# )

# 可选：精度分析（放置模型输入节点所对应的图片）
# converter.set_do_accuracy_analysis(
#     accuracy_analysis_picture_list=['img1.jpg']
# )

# 执行转换
converter.convert(
    mean_rgb=[[123.675, 116.28, 103.53]],
    std_rgb=[[58.395, 57.12, 57.375]]
)

# 清理临时文件
converter.clean()
```


## 3.2 `OnnxToQNN` — ONNX 转 Qualcomm QNN
模仿示例的子模块，编写转换脚本``Your_Model_onnx2qnn.py``<br>
>之前优化后的 ONNX 模型文件放置在`models_convert/onnx/`目录下<br>
>转换后的 QNN 模型文件放置在`models_convert/qnn/`目录下<br>

```python
from pathlib import Path
from utilities.onnx_to_qnn import OnnxToQNN

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent

# 模型文件路径
MODEL_PATH = str(current_dir / 'models_convert/onnx/yourmodel.onnx')

# 导出路径
QNN_MODEL = str(current_dir / 'models_convert/qnn/yourmodel_i8.bin')

# 量化校准数据集路径（可选）
DATASET = str(parent_dir / 'datasets/datasets.txt')

TARGET_PLATFORM = 'qcs6490'


converter = OnnxToQNN(
    model_path=MODEL_PATH,
    qnn_model_path=QNN_MODEL,
    dataset_path=DATASET,
    target_platform='qcs6490'             # 'qcs6490' / 'qcs8550' / 'qcs9075'
)

# 可选：量化配置
converter.set_quantization_method(
    param_quant_method='entropy',    # 'min-max' / 'sqnr' / 'percentile' / 'mse' / 'entropy'
    act_quant_method='entropy',
    bitwidth='w8a8',                 # 'w4a8' / 'w4a16' / 'w8a8' / 'w8a16' / 'w16a16'
    bias_bitwidth=8                  # 8 或 32
)

# 可选：使用自定义校准数据（.raw 格式）
# converter.use_custom_alibration_data(
#     custom_alibration_data_path='path/to/calibration_data.txt'
# )

# 执行转换
converter.convert(
    mean_rgb=[[123.675, 116.28, 103.53]],
    std_rgb=[[58.395, 57.12, 57.375]],
    set_input_order='nhwc'           # 或 'nchw'
)

# 清理临时文件
converter.clean()
```

> **关于 `mean_rgb` 与 `std_rgb`（输入归一化参数）：**
> 
> 这两个参数定义了模型输入的正规化公式：**`normalized_input = (input - mean) / std`**
> 
> | 转换工具 | 归一化实现方式 |
> |---------|--------------|
> | **OnnxToRKNN** | 通过 `rknn.config(mean_values=..., std_values=...)` 将归一化参数写入 RKNN 模型配置，由 **NPU 硬件在推理时自动完成**，不修改 ONNX 图 |
> | **OnnxToQNN** | 直接在 ONNX 模型图中插入 **Sub（减均值）和 Div（除标准差）节点**，将归一化烘焙进模型结构后再转换 |
> 
> - 每个内层列表对应一个模型**输入**的 RGB 三通道值（多输入模型需提供多个列表）
> - 默认值 `mean=[[0,0,0]]`, `std=[[1,1,1]]` 表示不做归一化
> - 修改后模型的输入数值范围变化: 
>   - e.g. Yolo. set mean=0, std=255: float[0, 1] -> float/uint8[0, 255]
>   - e.g. RetinaFace_mobile. set mean=0, std=1: float[0, 255] -> float/uint8[0, 255]
>   - e.g. AVTrack. set mean = [0.485x255, 0.456x255, 0.406x255], std = [0.229x255, 0.224x255, 0.225x255]: float[0, 1] -> float/uint8[0, 255]
> - 叶姐姐🍃: 建议在整数量化部署的时候通过设置 `mean` 和 `std` 来使得模型的输入范围匹配你要输入数据的范围。比如如小女子都是输入图像的，自然就把模型的输入范围设置成和图像的像素值范围一致，即 0-255。


**量化算法对比：**

| 算法 | 算法全称 | 适用场景 | 相对量化速度 | 相对精度 | 相对量化内存用量 |
|------|---------|---------|---------|---------|-------------|
| `normal(min-max)` | 最小-最大归一化量化 | 基础均匀量化；<br>权重分布对称、无极端离群值 | ⭐⭐⭐⭐⭐<br>(最快) | ⭐⭐⭐<br>(基准) | 💾<br>(仅需2个标量) |
| `kl_divergence` | KL散度分布拟合量化 | 激活值分布非均匀/长尾；<br>需保留原始分布形状的敏感模型；精度优先场景 | ⭐⭐<br>(较慢) | ⭐⭐⭐⭐⭐<br>(最高) | 💾💾💾💾💾<br>(直方图+多轮迭代) |
| `mmse` | 最小均方误差量化 | 对量化误差高度敏感的模型；<br>高精度分类/检测任务；需平衡截断与舍入误差 | ⭐⭐⭐<br>(中等) | ⭐⭐⭐⭐<br>(较高) | 💾💾💾💾<br>(需迭代+激活缓存) |
| `sqnr` | 信噪比优化量化 | 信噪比敏感场景（音频/信号处理）；<br>需量化层敏感度分析；混合精度选择 | ⭐⭐⭐⭐<br>(较快) | ⭐⭐⭐⭐<br>(较高) | 💾💾💾<br>(需缓存统计量) |
| `percentile` | 百分位截断量化 | 存在少量离群值的激活值场景；<br>需平衡鲁棒性与精度；移动端部署场景 | ⭐⭐⭐⭐<br>(较快) | ⭐⭐⭐⭐<br>(较高) | 💾💾<br>(需排序缓冲) |
| `entropy` | 信息熵校准量化 | 分布复杂/多峰；<br>需最小化信息损失的深度学习模型；大模型量化校准 | ⭐⭐<br>(较慢) | ⭐⭐⭐⭐⭐<br>(最高) | 💾💾💾💾💾<br>(直方图+熵计算+迭代) |

> 注：rknn的量化工具是单线程的，所以什么量化算法都较慢。而qnn的量化工具是多线程的，所以什么量化算法都较快


## 4. 量化精度分析

转换完成后，可通过精度分析定位量化损失最大的层，指导混合量化优化。详见 **[📊 量化精度分析指南](./ACCURACY_ANALYSIS_TOOLUSE.md)**。


## `GenYoloCroppedDataset` — 基于 YOLO 检测的裁剪数据集生成

### 使用场景
部分多输入模型（如目标跟踪、人脸/物体识别）的工作场景是：一个输入为**完整原图**，另一个输入为**裁剪后的感兴趣区域（ROI）**。如果用未裁剪的图像做量化校准，校准数据与真实推理场景不匹配，会导致量化精度下降。

`GenYoloCroppedDataset` 通过内置的 YOLO 检测模型自动识别并裁剪感兴趣区域，生成与原图一一配对的裁剪数据集，用于量化校准。

### 基础用法

```python
from utilities.yolo_cropped_dataset_gen import GenYoloCroppedDataset

generator = GenYoloCroppedDataset(
    dataset_path='datasets/datasets.txt',   # 原始数据集索引文件
    output_dir_name='cropped_images'         # 裁剪图片输出目录名
)

# 生成裁剪数据集（返回索引文件路径）
dataset_list_path = generator.generate()

# 可选：清理临时文件（输入副本、裁剪图片、索引文件）
# generator.clean()
```

### 进阶用法：配合另一个 AI 模型做后处理

如果目标模型需要先经过另一个 AI（如跟踪模型的 backbone）再输入，可以用 `set_postprocess_by_another_model()` 让裁剪后的图片自动通过该模型推理，输出 `.npy` 或 `.raw` 格式的 Tensor 数据：

```python
generator = GenYoloCroppedDataset(
    dataset_path='datasets/datasets.txt',
    output_dir_name='cropped_images'
)

# 指定一个或多个模型，对裁剪后的图片做推理，替换为推理输出
generator.set_postprocess_by_another_model(
    another_model_path_and_target_list=[
        ('path/to/model_T.onnx', 'input'),   # 对完整图（input 侧）推理
        ('path/to/model_S.onnx', 'output'),  # 对裁剪图（output 侧）推理
    ],
    output_shape='nchw',       # 输出张量布局
    outpur_format='.npy'       # 输出文件格式
)

dataset_list_path = generator.generate()

# 可选：清理临时文件（输入副本、裁剪图片、索引文件）
# generator.clean()
```

> **注意：** 生成的临时文件位于 `utilities/tmp/` 目录下，使用完毕后再调用 `generator.clean()` 清理。


### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_path` | `str` | — | 原始数据集索引文件路径（每行一个图片路径） |
| `output_dir_name` | `str` | `'cropped_images'` | 裁剪图片存放目录名（在 `utilities/tmp/` 下创建） |
| `swap_image_pair` | `bool` | `False` | 输出文件中是否交换配对顺序（`True` → `裁剪图 完整图`） |
| `another_model_path_and_target_list` | `list[tuple[str, str]]` | `None` | 后处理模型列表，每项为 `(模型路径, 'input'\|'output')` |
| `output_shape` | `str` | `'chw'` | 推理输出张量布局：`'chw'` / `'hwc'` / `'nchw'` / `'nhwc'` |
| `outpur_format` | `str` | `'.npy'` | 推理输出文件格式：`'.npy'` / `'.raw'` |

### `generate()` 返回值

返回配对数据集索引文件的路径字符串（`.txt`），每行格式：

```
/path/to/full_image.jpg   /path/to/cropped_image.jpg
```

若 `swap_image_pair=True`，顺序反转为 `裁剪图 完整图`；若配置了后处理模型，对应侧的路径会替换为推理输出文件路径。


