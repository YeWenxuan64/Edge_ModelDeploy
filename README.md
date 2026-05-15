# Edge Model Deploy

## 概述

本项目提供一套**可复用的模型转换工具链**，覆盖从训练框架（PyTroch, TensorFlow）的cv模型统一转换为 ONNX，再量化部署到边缘端 NPU（Rockchip RKNN / Qualcomm QNN）
> 同时本项目也是本小姐🍃的项目[Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder)的模型部署部分喵~


| 转换阶段 | 工具 | 说明 |
|---------|------|------|
| **PyTorch/TensorFlow → ONNX** | 各子模块独立脚本 | 处理算子兼容、动态图固化、模型优化 |
| **ONNX → RKNN** | `utilities/onnx_to_rknn.py` | Rockchip NPU（RK3588 / RK3576），支持 INT8 量化、混合量化|
| **ONNX → QNN** | `utilities/onnx_to_qnn.py` | Qualcomm NPU（HTP），支持多精度量化 |
| **数据集生成** | `utilities/yolo_croped_dataset_gen.py` | 基于 YOLO 检测自动裁剪量化校准数据集 |

> **工具链是核心资产** — 每个子模块（AVTrack、NanoTrackV3、RetinaFace 等）都复用同一套 `utilities/` 转换工具，只需编写模型特有的 PyTorch → ONNX 导出脚本即可。


## 项目结构

```
Focus-Finder_ModelDeploy/
├── utilities/                       # ⭐ 共享转换工具链（核心）
│   ├── onnx_to_rknn.py              # ONNX → RKNN 转换（Rockchip）
│   ├── onnx_to_qnn.py               # ONNX → QNN 转换（Qualcomm）
│   ├── yolo_croped_dataset_gen.py      # YOLO 检测辅助的量化数据集生成
│   ├── utils.py                     # 通用工具函数
│   └── qairt/                       # Qualcomm AI Runtime SDK # 需自行下载并放入
├── datasets/                        # 量化校准数据集
│   ├── datasets.txt                 # 通用数据集索引           # 可自行挑选和编写
│   ├── datasets_face.txt            # 人脸数据集索引           # 可自行挑选和编写
│   └── ...                          # 数据集图片文件夹         # 需自行下载和挑选
├── requirements.txt                 # Python 依赖
├── AVTrack_ModelDeploy/             # git submodule → AVTrack 跟踪模型
├── RetinaFace-mobile_ModelDeploy/   # git submodule → RetinaFace 人脸检测
├── NanoTrackV3_ModelDeploy/         # git submodule → NanoTrackV3 跟踪模型
├── MSI-Net_ModelDeploy/             # git submodule → MSI-Net 图像融合
├── Yolo11_ModelDeploy/              # git submodule → YOLO11 检测
└── Yolo26_ModelDeploy/              # git submodule → YOLO26 检测
```

## 示例模型

本项目以 git submodule 管理多个模型部署模块：

| 子模块 | 模型类型 | 来源 |
|--------|---------|------------|
| [AVTrack_ModelDeploy](./AVTrack_ModelDeploy/)                     | 视觉目标跟踪 | ICML 2024 — *Learning Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking* |
| [RetinaFace-mobile_ModelDeploy](./RetinaFace-mobile_ModelDeploy/) | 人脸检测     | RetinaFace 轻量化版本 |
| [NanoTrackV3_ModelDeploy](./NanoTrackV3_ModelDeploy/)             | 视觉目标跟踪 | NanoTrack 系列 |
| [MSI-Net_ModelDeploy](./MSI-Net_ModelDeploy/)                     | 显著性检测   | MSI-Net<br>Neural Networks — *Contextual encoder-decoder network for visual saliency prediction* |
| [Yolo11_ModelDeploy](./Yolo11_ModelDeploy/)                       | 物体检测     | ultralytics-YOLO11(RKNN custom-made) |
| [Yolo26_ModelDeploy](./Yolo26_ModelDeploy/)                       | 物体检测     | ultralytics-YOLO26 |

每个子模块独立维护，包含该模型的完整转换流程与预训练权重。


## 快速开始

### 1. 克隆项目（含子模块）

```bash
git clone --recurse-submodules https://github.com/YeWenxuan64/Focus-Finder_ModelDeploy.git
cd Focus-Finder_ModelDeploy
```

### 2. 安装依赖
> **假如你在使用python虚拟环境，请在你正在使用的环境下安装**<br>
> **RKNN 转换工作流支持Python 3.10 - 3.12，QNN 转换工作流仅支持Python 3.10**
```bash
pip install -r requirements.txt
```

**如果需要转换为QNN模型，则需要下载高通的`QAIRT SDK`，解压并放置在`onnx_to_qnn.py`同级目录下**
```
Eedge_ModelDeploy/
├── utilities/
│   ├── onnx_to_rknn.py
│   ├── onnx_to_qnn.py
│   ├── yolo_det_dataset_gen.py
│   ├── utils.py
│   └── qairt/                       # Qualcomm AI Runtime SDK # 需自行下载并放入
│       └── 2.38.0.250901/           # SDK 版本                # 可自行挑选
│           ├── bin/
│           └── ...
└── ...                              
```

### 2.5 下载量化校准数据集

量化转换需要校准数据集，用于量化过程中根据模型对于输入数据的反应来计算量化参数<br>
如果项目中没有预置数据集，可以从以下官方来源下载：

| 数据集 | 用途 | 官方下载链接 |
|--------|------|------------|
| **COCO 2017 val** | 通用目标检测/跟踪模型校准 | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (5GB) |
| **COCO 2017 train** | 更多校准样本（可选） | [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) (19GB) |
| **WIDER Face** | 人脸检测模型校准 | [WIDER Face Images](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/wider_face_split/wider_face_split.zip) + [WIDER Face Validation](https://huggingface.co/datasets/Wilder/WIDERFace/resolve/main/wider_face_split.zip) |
| **ImageNet** | 通用分类/检测模型校准（需注册） | [ImageNet 官网](https://www.image-net.org/download) 或 [Small ImageNet](https://www.image-net.org/small/download.php) |

> **提示：** <br>
> 校准数据集不需要太大，通常 **50~500 张**具有代表性的图片即可达到良好量化精度<br>
> 校准数据集推荐使用**符合模型应用场景**的通用数据集，比如自己训练的模型则需要使用自己的数据集<br>
> 格式为每行一个图片路径的 `.txt` 文件。若模型有多个输入，则每行<输入个数>个图片路径<br>


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

### 3. 使用转换工具

每个子模块的转换流程一致，以 Yolo26_ModelDeploy 为例：

```bash
# 1.PyTorch → ONNX（各模型独立实现）
cd Yolo26_ModelDeploy
python Yolo26_pytorch2onnx.py

# 2.1.ONNX → RKNN（复用 utilities）
python Yolo26_onnx2rknn.py
# 输出到: Yolo26_ModelDeploy/models_convert/rknn

# 2.2.ONNX → QNN（复用 utilities）
python Yolo26_onnx2qnn.py
# 输出到: Yolo26_ModelDeploy/models_convert/qnn
```


## 工具链使用指南
[工具链使用指南](./README_TOOLUSE.md)


## 兼容性

### 硬件兼容性

| 目标平台        | AI处理器    | 芯片     | 转换工具 | 量化格式 |
|----------------|-------------|---------|---------|---------|
| Rockchip       | NPU         | RK3588 RK3576 RK3566 | `onnx_to_rknn.py` | INT8 / FP16 / 混合量化 |
| Qualcomm (HTP) | Hexagon DSP | QCS6490 | `onnx_to_qnn.py` | INT8 / INT4 / FP16 |

### 软件兼容性
> 目前本工具仅支持**计算机视觉（CV）**类的模型部署<br>
> 目前本工具仅支持固定输入、输出尺寸的模型<br>


## License

MIT License — Copyright (c) 2026 叶文轩

各子模块的模型、代码，以及数据集遵循其原始 License。
