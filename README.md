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

⚠️Pre-Release Warning⚠️

</div>

## 📖 概述

本项目提供一套**可复用的模型转换工具链**与**精度分析工具**：把训练框架（PyTorch / TensorFlow）的 CV 模型统一转换为 ONNX，再量化部署到边缘端 NPU（Rockchip RKNN / Qualcomm QNN）。
> 同时本项目也是本小姐🍃的项目[Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder)的模型部署部分喵~

> **工具链是核心资产** — 每个模型子模块只需编写该模型特有的 PyTorch → ONNX 导出脚本，量化、混合精度与精度分析全部复用同一套 `utilities/` 转换工具。

| 转换阶段 | 工具 | 运行平台 | 产物 |
|---------|------|---------|------|
| **PyTorch/TensorFlow → ONNX** | 各子模块独立脚本 | Windows / Linux（Python 3.10+） | `.onnx` |
| **ONNX → RKNN** | `utilities/onnx_to_rknn.py` | 仅 Linux（Python 3.10 - 3.12） | `.rknn` |
| **ONNX → QNN** | `utilities/onnx_to_qnn.py` | Windows / Linux（Python 3.10） | `.bin`（HTP context binary） |
| **ONNX → QDQ-ONNX（AIMET）** | `utilities/onnx_aimet_quant.py` | 仅 Linux（Python 3.10） | QDQ `.onnx` + encodings |
| **校准数据集生成** | `utilities/dataset_preprocess.py` | Windows / Linux | 裁剪 / 推理后的数据集索引 `.txt` |
| **量化精度分析** | `utilities/accuracy_debugger.py` | 随对应转换工具 | 统计图 `.png` + 计算图 `.html` + 精度 `.csv` |


## 🧪 支持矩阵

| 目标平台 | AI 处理器 | 芯片 | 量化格式 | 逐层（single）精度分析 |
|---------|----------|------|---------|---------------------|
| Rockchip | NPU | RK3588 / RK3576 / RK3566 ... | INT8 / FP16 / 混合量化（子图 FP16） | 由 RKNN Toolkit2 内置 `rknn.accuracy_analysis()` 提供 |
| Qualcomm (HTP) | Hexagon NPU | QCS6490 / QCS8550 / QCS9075 / SC8280X ... | INT8 / INT4 / FP16 / 混合量化（子图 w8a16 / w16a16 / FP16 / FP32） | Linux 与 Windows ARM64（WoA）可用；Windows x86 无 HTP 后端，自动退化为仅累计（entire）分析 |

> - 位宽、量化算法、混合量化的具体参数见 [工具链使用指南](./docs/TOOLUSE.md)，精度指标含义见 [量化精度分析指南](./docs/ACCURACY_ANALYSIS.md)。
> - 混合量化约束：`w16a16` 需 `dsp_arch ≥ v73`（qcs8550 / qcs9075）；FP16 区域需 SoC 支持 HTP float16 运算（实测 qcs6490 / v68 不支持）。
> - 高通平台支持更多芯片，只是暂时没写上去。可以提issues附上想要硬件信息，小女子可以适配喵~ 🐾


## 📦 安装

### 克隆本仓库

```bash
git clone https://github.com/YeWenxuan64/Edge_ModelDeploy.git
cd Edge_ModelDeploy
```

> 各模型的部署工程是**独立仓库**，不随本仓库附带；跑示例见下文 [跑通第一个模型](#跑通第一个模型)。

### 系统依赖
```bash
sudo apt-get update

# 如果你在VMware虚拟机
# sudo apt-get install open-vm-tools open-vm-tools-desktop

sudo apt-get install cmake git
sudo apt install python3-pip python3-venv python3-tk

```

### Python 依赖

> 假如你在使用python虚拟环境，请在**你正在使用的**环境下安装<br>
> - 附上虚拟环境的创建和进入:
> ```bash
> python3 -m venv ~/python_venv
> source ~/python_venv/bin/activate
> ```

**安装顺序至关重要**，请严格按以下顺序逐条执行，**不要加 `--upgrade` 标志**：

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1. | `pip install -r requirements_cpu.txt` | 基础依赖与 PyTorch / TensorFlow **CPU 版** |
| 2. | `pip install -r requirements_overwrite.txt` | **包版本覆盖** — 将某些包降级/锁定到兼容版本（必须在最后装！） |
| 3. | `pip install rknn-toolkit2 --no-deps` | Rockchip RKNN 工具（可选，`--no-deps` 避免上游依赖冲突） |
| 4. | `pip install aimet-onnx` | Qualcomm AIMET 量化工具（可选） |

<details>
<summary>为什么安装顺序 / CPU 版框架很重要？</summary>

核心原因是 **`protobuf`** 和 **`easydict`** 等包在 TensorFlow 与 PyTorch 不同版本间存在冲突 — 先装 A 框架拉高版本，再装 B 框架时可能不兼容。`requirements_overwrite.txt` 会在最后统一锁定兼容版本。此外 **RKNN / QNN 框架对 TensorFlow 和 PyTorch 的版本要求极其严苛**，混装极易踩坑，分步安装可以精确控制每一层的依赖版本。

建议安装 **CPU 版**的神经网络框架：GPU 版体积会稍微**膨胀**，且 Linux 下会强制安装体积巨大的 Nvidia 驱动。
</details>

### QAIRT SDK（转 QNN 才需要）

如果需要转换为 QNN 模型，则需要下载高通的 `QAIRT SDK`，解压并放置在 `utilities/qairt/` 目录下（不入库，需自行下载）<br>
网页下载: [Qualcomm AI Runtime SDK](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community?osArch=Any&osType=All&version=2.38.0.250901)<br>
链接下载: [Qualcomm_AI_Runtime_SDK_2.38.0.250901.zip](https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.38.0.250901/v2.38.0.250901.zip)

> - 转换模型所使用的 SDK 版本建议**低于等于**推理时所用的 SDK 版本。
> - 工具会自动选取 `utilities/qairt/` 下**字典序最大**的目录作为当前使用的 SDK 版本；本文档与示例默认按 `2.38.0.250901` 编写。

```
Edge_ModelDeploy/
└── utilities/
    ├── onnx_to_qnn.py
    └── qairt/                       # Qualcomm AI Runtime SDK # 需自行下载并放入
        └── 2.38.0.250901/           # SDK 版本                # 当前被自动选中（字典序最大）
            ├── bin/
            └── ...
```

### 量化校准数据集

量化转换需要校准数据集，用于量化过程中根据模型对于输入数据的反应来计算量化参数<br>

> ✅ **仓库已预置部分常用数据集**，位于 `datasets/` 下，可直接使用、**无需下载**：
>
> | 数据集文件夹 | 对应索引文件 | 适用场景 |
> |------------|------------|---------|
> | `datasets/COCO2017_subset/` | `datasets.txt` / `datasets_full.txt` / `datasets_short.txt` | 通用目标检测/跟踪 |
> | `datasets/WIDER_val_subset/` | `datasets_face.txt` | 人脸检测 |
> | `datasets/saliency/` | `datasets_saliency.txt` | 显著性检测 |
> | `datasets/ILSVRC2012_img_val_samples/` | （无索引，可自建） | 通用分类 |
>
> ⚠️ **自训练模型请自行准备校准数据集** — 预置数据集面向通用场景，若你的模型是针对特定领域/场景训练的，校准数据应**贴近模型的实际应用场景**，否则量化精度可能不理想。

如果项目中没有预置数据集（或需要扩充），可以从以下官方来源下载：

| 数据集 | 用途 | 官方下载链接 |
|--------|------|------------|
| **COCO 2017 val** | 通用目标检测/跟踪模型校准 | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (~5GB) |
| **WIDER Face** | 人脸检测模型校准 | [WIDER Face 官网](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace) (~400MB) |
| **ImageNet** | 通用分类/检测模型校准（需注册） | [ImageNet 官网](https://www.image-net.org/download) 或 [Small ImageNet](https://www.image-net.org/small/download.php) |

> **提示：** <br>
> 校准数据集不需要太大，通常 **50~200 张**具有代表性的图片即可达到良好量化精度<br>
> 校准数据集推荐使用**符合模型应用场景**的通用数据集，比如自己训练的模型则需要使用自己的数据集<br>
> 格式为每行一个图片路径的 `.txt` 文件。若模型有多个输入，则每行<输入个数>个图片路径<br>
> 可以用 `utilities/utils.py` 的 `collect_image_paths()` 自动扫描目录生成索引文件（见 [TOOLUSE.md](./docs/TOOLUSE.md#0-安装依赖与准备量化校准数据集)）


## 🚀 跑通第一个模型

前置条件：**Python 依赖已装好**；转 QNN 还需把 QAIRT SDK 放进 `utilities/qairt/`，转 RKNN 需 Linux + `rknn-toolkit2`。

示例工程 `yolo26_ModelDeploy` 是独立仓库，**必须放在本仓库根目录下，与 `utilities/`、`datasets/` 同级**：

```bash
# 在 Edge_ModelDeploy/ 根目录执行
git clone https://github.com/YeWenxuan64/yolo26_ModelDeploy.git
```

```
Edge_ModelDeploy/
├── utilities/                    # 共享转换工具链
├── datasets/                     # 校准数据集与索引 txt
└── yolo26_ModelDeploy/           # ← 模型工程 clone 到这一层
```

> 转换脚本用 `Path(__file__).parent.parent` 反推仓库根，再去 `import utilities` 和读取 `datasets/*.txt`：**目录层级放错就会 ImportError 或找不到校准数据集**；只要放对位置，从任意工作目录执行都能跑通（产物路径同样按脚本自身位置解析，不依赖 cwd）。

然后先导出 ONNX，再按目标 NPU 二选一执行对应脚本：

```bash
cd yolo26_ModelDeploy

# 1. PyTorch -> ONNX（权重获取、算子兼容、动态图固化）
python Yolo26_pytorch2onnx.py

# 2.1 ONNX -> RKNN          # 产物: yolo26_ModelDeploy/models_convert/rknn/*.rknn
python Yolo26_onnx2rknn.py

# 2.2 ONNX -> QNN（HTP）    # 产物: yolo26_ModelDeploy/models_convert/qnn/*.bin
python Yolo26_onnx2qnn.py
```

转换脚本会逐步打印使用的量化配置与产物路径；中间产物统一落在 `utilities/tmp/<模型名>_to_rknn/` 或 `..._to_qnn/`，脚本末尾的 `converter.clean()` 会清理它们，**需要保留精度分析结果时在 `clean()` 前取走**。

- 想同时看量化损失落在哪一层：在该工程脚本里 `convert()` **之前**调用 `converter.set_do_accuracy_analysis(accuracy_analysis_picture_list=['img.jpg'])`，图表解读见 [量化精度分析指南](./docs/ACCURACY_ANALYSIS.md)。
- 想换成自己的模型：新建 `Your_ModelDeploy/` 工程目录、套用同样的三个脚本模板，步骤见 [工具链使用指南](./docs/TOOLUSE.md#1-创建转换工程)。


## 🔀 模型转换工作流

```
                                   ┌────────────────────────────────┐
                                   │      PyTorch / TensorFlow      │
                                   │   (Training Framework Model)   │
                                   └───────────────┬────────────────┘
                                                   │  Per-submodule scripts
                                   ┌───────────────▼────────────────┐
                                   │             ONNX               │
                                   │ (Intermediate Representation)  │
                                   └───────────────┬────────────────┘
                                                   │
                ┌──────────────────────────────────┼──────────────────────────────────┐
                │                                  │                                  │
┌───────────────▼────────────────┐ ┌───────────────▼────────────────┐ ┌───────────────▼────────────────┐
│           OnnxToRKNN           │ │           OnnxToQNN            │ │       AimetOnnxQuantizer       │
│  (utilities/onnx_to_rknn.py)   │ │  (utilities/onnx_to_qnn.py)    │ │ (utilities/onnx_aimet_quant.py)│
├────────────────────────────────┤ ├────────────────────────────────┤ ├────────────────────────────────┤
│ 1. rknn.config()               │ │ 1. run_env_script()            │ │ 1. QuantizationSimModel        │
│    Config quant algorithm      │ │    Source QAIRT SDK env vars   │ │    param/act qtype + scheme    │
│                                │ │                                │ │                                │
│ 2. rknn.load_onnx()            │ │ 2. modify_onnx_model()         │ │ 2. set_tensor_precision()      │
│    Load ONNX model             │ │    Add normalization Conv node │ │    (optional) mixed precision  │
│                                │ │    Reorder nodes by I/O        │ │                                │
│ 3. rknn.build()                │ │                                │ │ 3. compute_encodings()         │
│    ├─ With dataset -> INT8     │ │ 3. get_onnx_model_info()       │ │    Dataset calibration         │
│    ├─ No dataset  -> FP16      │ │    Parse input/output dims     │ │                                │
│    └─ Hybrid quant -> step1+2  │ │                                │ │ 4. to_onnx_qdq()               │
│                                │ │ 3.5 do_hybrid_quantization()   │ │    QDQ ONNX + encodings JSON   │
│ 4. rknn.export_rknn()          │ │    Hybrid overrides JSON       │ │        │             │         │
│    Export .rknn model file     │ │    16-bit / FP16 subgraph      │ │        │             │         │
│                                │ │    w16a16: extra 16/16 calib   │ │        │             │         │
│ 5. rknn.release()              │ │                                │ │        │             │         │
│    Release resources           │ │ 4. generate_calibration_data() │ │        │             │         │
│                                │ │    Read imgs -> Preprocess     │ │        │             │         │
│ 6. clean()                     │ │    -> .raw files               │ │        │             │         │
│    Clean temp files            │ │                                │ │        │             │         │
│                                │ │ 5. convert_onnx_model() <──────┼─┼────────┴─────────────┘         │
└───────────────┬────────────────┘ │    qairt-converter             │ └───────────────┬────────────────┘
                │                  │    ONNX -> DLC                 │                 │
                ▼                  │                                │                 ▼
┌────────────────────────────────┐ │ 6. quantize_model()            │ ┌────────────────────────────────┐
│         .rknn Model            │ │    qairt-quantizer             │ │      QDQ ONNX + encodings      │
│     Rockchip NPU Executable    │ │    DLC -> Quantized DLC        │ │        AIMET Quantized         │
│   (RK3588 / RK3576 / RK3566)   │ │                                │ │ -> ONNX Deploy / QNN external  │
└────────────────────────────────┘ │ 7. write_config_file()         │ └────────────────────────────────┘
                                   │    Generate HTP backend config │
                                   │    JSON                        │
                                   │                                │
                                   │ 8. generate_context_binary()   │
                                   │   qnn-context-binary-generator │
                                   │    Quantized DLC -> .bin (HTP) │
                                   │                                │
                                   │ 9. clean()                     │
                                   │    Clean temp files            │
                                   └───────────────┬────────────────┘
                                                   │
                                                   ▼
                                   ┌────────────────────────────────┐
                                   │          .bin Model            │
                                   │     Qualcomm HTP Executable    │
                                   │ (QCS6490 / QCS8550 / QCS9075)  │
                                   └────────────────────────────────┘
```

> - 图中步骤号即各转换类 `convert()` 内部的调用顺序；QNN 步骤 8 的方法名为 `generate_context_binary_model()`。
> - **校准数据（步骤 4）先于 DLC 转换（步骤 5）**：混合量化的 `w16a16` 会用同一份校准数据先跑一遍全图 16/16 参考校准（框内 3.5），因此校准数据要提前生成。
> - 未提供校准数据集时跳过步骤 6，且步骤 5 的 `qairt-converter` 会自动带 `--float_bitwidth 16`，产物为 FP16 DLC。
> - `set_use_aimet()` 时 AIMET 取代步骤 4/6：由 AIMET 产出量化编码，经步骤 5 直接转成量化 DLC，子图混合精度走 AIMET 的 `set_tensor_precision()`。
> - 量化精度分析不在此图内，见 [量化精度分析指南](./docs/ACCURACY_ANALYSIS.md)。


## 🏗️ 项目结构

```
Edge_ModelDeploy/
├── utilities/                       # ⭐ 共享转换工具链（核心）
│   ├── onnx_to_rknn.py              # ONNX -> RKNN 转换（Rockchip）
│   ├── onnx_to_qnn.py               # ONNX -> QNN 转换（Qualcomm）
│   ├── onnx_aimet_quant.py          # AIMET 2.x 后训练量化（PTQ）-> QDQ ONNX + encodings
│   ├── dataset_preprocess.py        # 量化数据集预处理（YOLO 裁剪 / 模型批处理）
│   ├── accuracy_debugger.py         # 量化精度分析（逐层对比 + 可视化计算图，RKNN/QNN 共用）
│   ├── utils.py                     # 通用工具函数
│   ├── qairt/                       # Qualcomm AI Runtime SDK # 需自行下载放入
│   └── tmp/                         # 转换与分析的临时工作目录（转换时生成）
├── datasets/                        # 预置量化校准数据集 + 索引 txt
├── docs/                            # 工具链使用指南 / 精度分析指南 + 示例输出图
├── requirements_cpu.txt             # 基础依赖 + PyTorch / TensorFlow CPU 版
├── requirements_overwrite.txt       # 包版本覆盖（须最后安装）
└── <model>_ModelDeploy/             # 模型转换工程 # 独立仓库、不随克隆附带，需自行创建或 clone 到此处
```

每个 `<model>_ModelDeploy/` 的内部结构与命名约定见 [工具链使用指南](./docs/TOOLUSE.md#1-创建转换工程)。


## 🧠 示例模型

各模型的部署工程**都是独立仓库，不随本仓库克隆附带** —— `git clone` 下来只有 `utilities/` + `datasets/` + `docs/`，模型工程需按需单独 clone 到仓库根目录再使用。

| 示例模型工程 | 模型类型 | 来源 |
|------------|---------|------|
| [yolo26_ModelDeploy](https://github.com/YeWenxuan64/yolo26_ModelDeploy/) | 物体检测 | ultralytics-YOLO26 |

> 目前 (Pre-Release) 以 `yolo26_ModelDeploy` 作为示例模型喵~ 它自带该模型的预训练权重获取方式与完整转换脚本（`Yolo26_pytorch2onnx.py` / `Yolo26_onnx2rknn.py` / `Yolo26_onnx2qnn.py`），也是接入新模型时最直接的调用模板。


## 📚 工具链文档

| 文档 | 能查到什么 |
|------|-----------|
| [工具链使用指南](./docs/TOOLUSE.md) | 创建转换工程、`OnnxToRKNN` / `OnnxToQNN` / `AimetOnnxQuantizer` 的全部接口与参数、混合量化、数据集工具、`mean_rgb` / `std_rgb` 归一化语义、量化算法对比 |
| [量化精度分析指南](./docs/ACCURACY_ANALYSIS.md) | 精度指标含义、RKNN / QNN 两条路径的分析流程与输出文件、统计图与 Netron 风格计算图解读、分析之后的优化路径 |


## ⚠️ 局限性

> TODO — 小女子笨笨的，未来慢慢填坑喵~ 🐾

- [ ] **更多模型类型** — 目前仅支持 **计算机视觉（CV）** 类的模型部署，NLP / VLA / 语音等领域的模型暂不支持

- [ ] **动态尺寸** — 仅支持**固定输入、输出尺寸**的模型，动态 shape 的模型需要手动固定后再走转换流程

- [x] **QNN 混合量化** — 由于小女子太笨了，不会树和图数据结构，不会遍历计算图的特定节点来指定混合量化
    - 支持 `QAIRT` 原生的混合精度量化：可对**指定子图**使用 16-bit 整数量化（如 w16a16）或保留 FP16/FP32 浮点精度，其余部分仍按全局设置（默认 w8a8）量化为 INT8
    - 🛠️ 测试性支持（完成于 2026-08-12）
    - 🛠️ 重构于 2026-09-12

- [x] **QNN 高级量化（AIMET）** — 由于小女子太笨了，尚未接入 `AIMET (AI Model Efficiency Toolkit)` 的更高级量化方法
    - 支持基于 `AIMET (AI Model Efficiency Toolkit)` 的 **PQT** 量化方法，可作为独立的 QNNX 量化器或 QNN 的外置量化器使用
    - 🧪 实验性支持（完成于 2026-08-14）

- [x] **QNN 精度分析** — 由于小女子太笨了，QNN 的精度分析还不会用喵
    - 支持基于 `snpe-accuracy-debugger` 的精度分析，混合量化场景下自动使用纯浮点 DLC 作为 Golden 参考
    - 🛠️ 实验性支持（完成于 2026-06-22）（已弃用）
    - 支持基于 `qnn-net-run --debug` 的**双 DLC** 精度对比：FP32 Golden DLC（CPU 后端） vs 量化 DLC（Linux 走 HTP 后端；Windows x86 无 HTP 时退化为 CPU 后端），混合量化场景自动使用纯浮点 DLC 作为 Golden
    - 🛠️ 测试性支持（重构于 2026-09-03）
    - ✅ 预发布支持：支持 entire(累计) 和 single(逐层) 精度分析，可以运行在 X64/Arrch64-Linux, x64/Arm-Windows 


## 📄 License

[MIT License](./LICENSE) — Copyright (c) 2026 叶文轩

各子模块的模型、代码，以及数据集遵循其原始 License。
