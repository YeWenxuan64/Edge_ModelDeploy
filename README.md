<div align="center">

# Edge Model Deploy | 边缘模型部署器

![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=pink)


[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)]()
[![Edge AI](https://img.shields.io/badge/Edge-AI-purple)]()
[![ONNX](https://img.shields.io/badge/ONNX-1.0+-005CED?logo=onnx)](https://onnx.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![RKNN](https://img.shields.io/badge/Rockchip-RKNN-EC6F16)](https://github.com/airockchip/rknn-toolkit2)
[![QNN](https://img.shields.io/badge/Qualcomm-QNN%20(HTP)-31017D?logo=qualcomm)](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
[![AIMET](https://img.shields.io/badge/Qualcomm-AIMET-2853DC?logo=qualcomm)](https://github.com/quic/aimet)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

⚠️Pre-release Warning⚠️

</div>

## 📖 概述

本项目提供一套**可复用的模型转换工具链**，覆盖从训练框架（PyTroch, TensorFlow）的cv模型统一转换为 ONNX，再量化部署到边缘端 NPU（Rockchip RKNN / Qualcomm QNN）
> 同时本项目也是本小姐🍃的项目[Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder)的模型部署部分喵~


| 转换阶段 | 工具 | 说明 |
|---------|------|------|
| **PyTorch/TensorFlow → ONNX** | 各子模块独立脚本 | 处理算子兼容、动态图固化、模型优化 |
| **ONNX → RKNN** | `utilities/onnx_to_rknn.py` | Rockchip NPU（RK3588 / RK3576），支持 INT8 量化、混合精度量化|
| **ONNX → QNN** | `utilities/onnx_to_qnn.py` | Qualcomm NPU（HTP），支持 INT8/INT4 量化、混合精度量化 |
| **数据集生成** | `utilities/yolo_croped_dataset_gen.py` | 基于 YOLO 检测自动裁剪量化校准数据集 |

> **工具链是核心资产** — 每个子模块（AVTrack、NanoTrackV3、RetinaFace 等）都复用同一套 `utilities/` 转换工具，只需编写模型特有的 PyTorch → ONNX 导出脚本即可。


## 🏗️ 项目结构

```
Edge_ModelDeploy/
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
├── avtrack_ModelDeploy/             # git submodule → AVTrack 跟踪模型
├── retinaface_mobile_ModelDeploy/   # git submodule → RetinaFace 人脸检测
├── nanotrack_v3_ModelDeploy/        # git submodule → NanoTrackV3 跟踪模型
├── msi_net_ModelDeploy/             # git submodule → MSI-Net 图像融合
├── yolo11_ModelDeploy/              # git submodule → YOLO11 检测
└── yolo26_ModelDeploy/              # git submodule → YOLO26 检测
```

## 🧠 示例模型

本项目可用于本小姐的多个模型部署模块：

| 子模块 | 模型类型 | 来源 |
|--------|---------|------------|
| [avtrack_ModelDeploy](https://github.com/YeWenxuan64/avtrack_ModelDeploy/)                     | 视觉目标跟踪 | ICML 2024 — *Learning Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking* |
| [retinaface_mobile_ModelDeploy](https://github.com/YeWenxuan64/retinaface_mobile_ModelDeploy/) | 人脸检测     | RetinaFace 轻量化版本 |
| [nanotrack_v3_ModelDeploy](https://github.com/YeWenxuan64/nanotrack_v3_ModelDeploy/)           | 视觉目标跟踪 | NanoTrack 系列 |
| [msi_net_ModelDeploy](https://github.com/YeWenxuan64/msi_net_ModelDeploy/)                     | 显著性检测   | MSI-Net<br>Neural Networks — *Contextual encoder-decoder network for visual saliency prediction* |
| [yolo11_ModelDeploy](https://github.com/YeWenxuan64/yolo11_ModelDeploy/)                       | 物体检测     | ultralytics-YOLO11(RKNN custom-made) |
| [yolo26_ModelDeploy](https://github.com/YeWenxuan64/yolo26_ModelDeploy/)                       | 物体检测     | ultralytics-YOLO26 |

每个子模块独立维护，包含该模型的预训练权重获取方式与完整转换流程


## 📦 工具链部署
### 0. 环境要求

| 工作流           | Windows   | Linux | Python 版本 |
|-----------------|------------|------|-------------|
| 训练框架 to ONNX | 支持       | 支持 | 3.10+ |
| ONNX to RKNN    | 支持       | 支持 | 3.10 - 3.12 |
| ONNX to QNN     | 目前不支持 | 支持 | 3.10 |


### 1. 克隆项目（含子模块）

```bash
git clone --recurse-submodules https://github.com/YeWenxuan64/Edge_ModelDeploy.git
cd Edge_ModelDeploy
```

### 2. 安装依赖
#### 系统依赖
```bash
sudo apt-get update

# 如果你在VMware虚拟机
# sudo apt-get install open-vm-tools open-vm-tools-desktop

sudo apt-get install cmake git
sudo apt install python3-pip python3-venv python3-tk

```

#### Python 依赖

> 假如你在使用python虚拟环境，请在**你正在使用的**环境下安装<br>
> - 附上虚拟环境的创建和进入:
> ```bash
> python3 -m venv ~/python_venv
> source ~/python_venv/bin/activate
> ```

```bash
# 严格按此顺序执行（逐条！）
pip install -r requirements_base.txt
pip install -r requirements_torch_cpu.txt --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_tensorflow.txt
pip install -r requirements_overwrite.txt
pip install rknn-toolkit2 --no-deps
```

> ⚠️ **安装顺序至关重要**，请严格按以下顺序逐条执行，**不要合并为一条命令**，**不要加 `--upgrade` 标志**：
>
> | 步骤 | 命令 | 说明 |
> |------|------|------|
> | 1. | `pip install -r requirements_base.txt` | 基础依赖（numpy, opencv-python 等） |
> | 2. | `pip install -r requirements_torch_cpu.txt --index-url https://download.pytorch.org/whl/cpu` | PyTorch **CPU 版**。也可以装普通 GPU 版（`requirements_torch.txt`），但 GPU 版体积异常膨胀，且对其他依赖的版本约束更严格，在依赖有冲突的环境中很难调通 |
> | 3. | `pip install -r requirements_tensorflow.txt` | TensorFlow（与 PyTorch 有共同依赖，需在 base 之后安装） |
> | 4. | `pip install -r requirements_overwrite.txt` | **包版本覆盖** — 将某些包降级/锁定到兼容版本（必须在最后装！） |
> | 5. | `pip install rknn-toolkit2 --no-deps` | Rockchip RKNN 工具（`--no-deps` 避免上游依赖冲突） |
>
> **为什么顺序重要？** 核心原因是 **`protobuf`** 和 **`easydict`** 等包在 TensorFlow 与 PyTorch 不同版本间存在冲突 — 先装 A 框架拉高版本，再装 B 框架时可能不兼容。`requirements_overwrite.txt` 会在最后统一锁定兼容版本。此外 **RKNN / QNN 框架对 TensorFlow 和 PyTorch 的版本要求极其严苛**，混装极易踩坑，分步安装可以精确控制每一层的依赖版本。

#### QNN 依赖

**如果需要转换为QNN模型，则需要下载高通的`QAIRT SDK`，解压并放置在`onnx_to_qnn.py`同级目录下**<br>
网页下载: [Qualcomm AI Runtime SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community?osArch=Any&osType=All&version=2.38.0.250901)<br>
链接下载: [Qualcomm_AI_Runtime_SDK_2.38.0.250901.zip](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.38.0.250901/v2.38.0.250901.zip)

> 转换模型所使用的SDK版本建议**低于等于**推理时所用的SDK版本

```
Eedge_ModelDeploy/
├── utilities/
│   ├── onnx_to_rknn.py
│   ├── onnx_to_qnn.py
│   ├── yolo_croped_dataset_gen.py
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
| **COCO 2017 val** | 通用目标检测/跟踪模型校准 | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (~5GB) |
| **WIDER Face** | 人脸检测模型校准 | [WIDER Face 官网](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace) (~400MB) |
| **ImageNet** | 通用分类/检测模型校准（需注册） | [ImageNet 官网](https://www.image-net.org/download) 或 [Small ImageNet](https://www.image-net.org/small/download.php) |

> **提示：** <br>
> 校准数据集不需要太大，通常 **50~200 张**具有代表性的图片即可达到良好量化精度<br>
> 校准数据集推荐使用**符合模型应用场景**的通用数据集，比如自己训练的模型则需要使用自己的数据集<br>
> 格式为每行一个图片路径的 `.txt` 文件。若模型有多个输入，则每行<输入个数>个图片路径<br>


### 3. 使用转换工具

每个子模块的转换流程一致，以 yolo26_ModelDeploy 为例：

```bash
git clone --recurse-submodules https://github.com/YeWenxuan64/yolo26_ModelDeploy.git
cd yolo26_ModelDeploy

# 1.PyTorch → ONNX（各模型独立实现）
python Yolo26_pytorch2onnx.py

# 2.1.ONNX → RKNN（复用 utilities）
python yolo26_onnx2rknn.py
# 输出到: yolo26_ModelDeploy/models_convert/rknn

# 2.2.ONNX → QNN（复用 utilities）
python yolo26_onnx2qnn.py
# 输出到: yolo26_ModelDeploy/models_convert/qnn
```


## 📚 工具链使用指南

[工具链使用指南](./docs/README_TOOLUSE.md)
[量化精度分析指南](./docs/ACCURACY_ANALYSIS_TOOLUSE.md)

### 模型转换工作流

```
                   ┌─────────────────────────────────┐
                   │      PyTorch / TensorFlow       │
                   │    (Training Framework Model)   │
                   └───────────────┬─────────────────┘
                                   │  Per-submodule scripts
                                   ▼
                   ┌─────────────────────────────────┐
                   │             ONNX                │
                   │   (Intermediate Representation) │
                   └───────────────┬─────────────────┘
                                   │
                ┌──────────────────┴─────────────────┐
                │                                    │
                ▼                                    ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│           OnnxToRKNN            │  │           OnnxToQNN             │
│  (utilities/onnx_to_rknn.py)    │  │  (utilities/onnx_to_qnn.py)     │
│                                 │  │                                 │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│                                 │  │                                 │
│  1. rknn.config()               │  │  1. run_env_script()            │
│     Config quant algorithm      │  │     Source QAIRT SDK env vars   │
│                                 │  │                                 │
│  2. rknn.load_onnx()            │  │  2. modify_onnx_model()         │
│     Load ONNX model             │  │     Add normalization nodes     │
│                                 │  │     (Sub/Div)                   │
│  3. rknn.build()                │  │     Reorder nodes by I/O        │
│     ├─ With dataset → INT8      │  │     ONNX validate + Shape infer │
│     ├─ No dataset  → FP16       │  │                                 │
│     └─ Hybrid quant → step1+2   │  │  3. get_onnx_model_info()       │
│                                 │  │     Parse input/output dims     │
│  4. rknn.export_rknn()          │  │                                 │
│     Export .rknn model file     │  │  3.5 do_hybrid_quantization()   │
│                                 │  │     Hybrid overrides JSON       │
│  5. rknn.release()              │  │     16-bit / FP16 subgraph      │
│     Release resources           │  │                                 │
│                                 │  │  4. convert_onnx_model()        │
│  6. clean()                     │  │     qairt-converter             │
│     Clean temp files            │  │     ONNX ──► DLC (unquantized)  │
│                                 │  │                                 │
│                                 │  │  5. generate_calibration_data() │
│                                 │  │     Read imgs → Preprocess →    │
│                                 │  │     .raw files                  │
│                                 │  │                                 │
│                                 │  │  6. quantize_model()            │
│                                 │  │     qairt-quantizer             │
│                                 │  │     DLC ──► Quantized DLC       │
│                                 │  │                                 │
│                                 │  │  7. write_config_file()         │
│                                 │  │     Generate HTP backend config │
│                                 │  │     JSON                        │
│                                 │  │                                 │
│                                 │  │  8. generate_context_binary()   │
│                                 │  │     qnn-context-binary-generator│
│                                 │  │     Quantized DLC ──► .bin (HTP)│
│                                 │  │                                 │
│                                 │  │  9. clean()                     │
│                                 │  │     Clean temp files            │
└───────────────┬─────────────────┘  └───────────────┬─────────────────┘
                │                                    │
                ▼                                    ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│         .rknn Model             │  │          .bin Model             │
│     Rockchip NPU Executable     │  │     Qualcomm HTP Executable     │
│   (RK3588 / RK3576 / RK3566)    │  │  (QCS6490 / QCS8550 / QCS9075)  │
└─────────────────────────────────┘  └─────────────────────────────────┘
```




## 🔌 兼容性

### 硬件兼容性

| 目标平台        | AI处理器    | 芯片     | 转换工具 | 量化格式 |
|----------------|-------------|---------|---------|---------|
| Rockchip       | NPU         | RK3588 RK3576 RK3566 | `onnx_to_rknn.py` | INT8 / FP16 / 混合量化 |
| Qualcomm (HTP) | Hexagon DSP | QCS6490 QCS8550 QCS9075 | `onnx_to_qnn.py` | INT8 / INT4 / FP16 / 混合量化 |

> 提issues时请附上想要硬件信息，小女子可以适配喵~ 🐾


### 局限性

> TODO — 小女子笨笨的，未来慢慢填坑喵~ 🐾

- [ ] **更多模型类型** — 目前仅支持 **计算机视觉（CV）** 类的模型部署，NLP / 语音等领域的模型暂不支持

- [ ] **动态尺寸** — 仅支持**固定输入、输出尺寸**的模型，动态 shape 的模型需要手动固定后再走转换流程

- [x] **QNN 混合量化** — 由于小女子太笨了，不会树和图数据结构，不会遍历计算图的特定节点来指定混合量化
    - 支持 `QAIRT` 原生的混合精度量化：可对**指定子图**使用 16-bit 整数量化（如 w16a16）或保留 FP16/FP32 浮点精度，其余部分仍按全局设置（默认 w8a8）量化为 INT8
    - 🛠️ 测试性支持（完成于 2026-08-12）

- [x] **QNN 高级量化（AIMET）** — 由于小女子太笨了，尚未接入 `AIMET (AI Model Efficiency Toolkit)` 的更高级量化方法
    - 支持基于 `AIMET (AI Model Efficiency Toolkit)` 的 **PQT** 量化方法，可作为独立的 QNNX 量化器或 QNN 的外置量化器使用
    - 🧪 实验性支持（完成于 2026-08-14）

- [x] **QNN 精度分析** — 由于小女子太笨了，QNN 的精度分析还不会用喵
    - 支持基于 `snpe-accuracy-debugger` 的精度分析，混合量化场景下自动使用纯浮点 DLC 作为 Golden 参考
    - 🛠️ 测试性支持（完成于 2026-06-22）

## 📄 License

[MIT License](./LICENSE) — Copyright (c) 2026 叶文轩

各子模块的模型、代码，以及数据集遵循其原始 License。
