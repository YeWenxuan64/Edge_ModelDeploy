# 量化精度分析指南

## 概述

模型量化（INT8/INT4）虽然能大幅降低模型体积和推理延迟，但不可避免地会引入精度损失。<br>
**精度分析**的作用是：对同一张输入图片，逐层对比 **浮点模型（Golden）** 与 **量化模型（Quantized）** 的中间张量输出，以定位量化损失最大的层，为后续优化（混合量化、调整量化算法、校准数据集优化等）提供数据依据。

本工具链为 **RKNN** 和 **QNN** 两条转换路径均提供了精度分析能力：

| 转换工具 | 精度分析引擎 | 在本工具链的构造 | 触发方式 |
|---------|------------|---------|---------|
| `OnnxToRKNN` | RKNN Toolkit2 内置的 `rknn.accuracy_analysis()` | 直接调用 RKNN Toolkit2 内置的 `rknn.accuracy_analysis()` | 调用转换工具示例的方法 `set_do_accuracy_analysis()` 并传入图片 |
| `OnnxToQNN` | QAIRT `snpe-accuracy-debugger` | 脚本多步骤调用 `snpe-accuracy-debugger` | 调用转换工具示例的方法 `set_do_accuracy_analysis()` 并传入图片 |

---

## 一、RKNN 精度分析

### 1.1 使用方法

在 `OnnxToRKNN` 转换脚本中，调用 `set_do_accuracy_analysis()` 传入用于精度分析的图片路径：

```python
from utilities.onnx_to_rknn import OnnxToRKNN

converter = OnnxToRKNN(
    model_path=MODEL_PATH,
    rknn_model_path=RKNN_MODEL,
    dataset_path=DATASET,
    target_platform='rk3588'
)

# 启用精度分析（传入一张或多张图片）
converter.set_do_accuracy_analysis(
    accuracy_analysis_picture_list=[
        '/path/to/test_image.jpg'
    ]
)

converter.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])
converter.clean()
```

> **注意：** 
> - 单输入模型传入一个图片路径；多输入模型按输入数量传入多个路径。
> - `accuracy_analysis_picture_list` 中的图片将**不被包含在量化校准数据中**，而是专门用于精度评测，以确保评估的独立性。

### 1.2 输出解读

RKNN 精度分析会在转换过程中输出**每层的余弦相似度（Cosine Similarity）**，格式如下：

```
W   cosine  :  0.999984
W   cosine  :  0.999997
W   cosine  :  0.999789    <-- 注意：某个层的相似度明显低于其他层
W   cosine  :  0.999990
...
W   node    :  /model.0/conv/Conv
W   node    :  /model.0/act/Sigmoid
...
```

- **余弦相似度 ≥ 0.99**：量化精度良好，该层无需特别关注
- **余弦相似度 < 0.99**：该层量化损失较大，建议对该层做**混合量化（FP16 保留精度，其余 INT8）**

### 1.3 混合量化优化

当精度分析发现特定层量化损失较大时，可使用混合量化将这些层保留为 FP16：

```python
# 对精度敏感的子图使用 FP16 量化
converter.do_hybrid_quantization(
    custom_hybrid=[
        ['敏感层的输入节点名', '敏感层的输出节点名'],
    ]
)
```

---

## 二、QNN 精度分析

### 2.1 使用方法

在 `OnnxToQNN` 转换脚本中，调用 `set_do_accuracy_analysis()` 传入用于精度分析的图片路径：

```python
from utilities.onnx_to_qnn import OnnxToQNN

converter = OnnxToQNN(
    model_path=MODEL_PATH,
    qnn_model_path=QNN_MODEL,
    dataset_path=DATASET,
    target_platform='qcs6490'
)

# 启用精度分析（传入一张或多张图片）
converter.set_do_accuracy_analysis(
    accuracy_analysis_picture_list=[
        str(parent_dir / 'datasets/test_image.jpg')
    ]
)

converter.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[1, 1, 1]])
```

### 2.2 分析流程

QNN 精度分析自动执行以下三步：

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: Golden 推理                                      │
│  使用 snpe-accuracy-debugger 在未量化的 DLC 模型上推理     │
│  得到各层浮点参考输出 (Golden Reference)                   │
├──────────────────────────────────────────────────────────┤
│  Step 2: Quantized 推理                                   │
│  使用 snpe-accuracy-debugger 在量化后的 DLC 模型上推理     │
│  得到各层量化输出 (Inference Results)                      │
├──────────────────────────────────────────────────────────┤
│  Step 3: Verification 对比                                │
│  逐层计算 CosineSimilarity + MSE                          │
│  生成 summary.csv 和可视化图表                             │
└──────────────────────────────────────────────────────────┘
```

### 2.3 输出文件

分析完成后，结果文件位于 `~/accuracy_analysis/`（用户主目录下）：

| 目录/文件 | 说明 |
|----------|------|
| `~/accuracy_analysis/golden_dir/` | Golden（浮点）模型推理结果 |
| `~/accuracy_analysis/quant_dir/` | Quantized（量化）模型推理结果 |
| `~/accuracy_analysis/verification/<timestamp>/summary.csv` | 逐层精度对比表 |
| `utilities/tmp/accuracy_analysis/latest/` | 最新分析结果的副本 |
| `utilities/tmp/accuracy_analysis_summary.png` | 可视化汇总图表 |

### 2.4 可视化图表解读

`accuracy_analysis_summary.png` 包含两个子图：

**上图 — 欧氏距离 (Euclidean Distance) & MSE（逐层柱状图 + 折线图）**
- 柱状图：每层输出与 Golden 之间的欧氏距离（L2 距离），值越大说明量化误差越大
- 蓝色折线（右轴）：每层的 MSE 值，用于辅助判断误差分布
- **重点关注柱子突出的层** — 它们是量化损失最大的层

**下图 — 余弦相似度 (Cosine Similarity)（逐层折线图）**
- 绿色折线：每层输出与 Golden 之间的余弦相似度
- 红色虚线：0.99 警戒阈值
- **低于红线（< 0.99）的层需要重点关注** — 考虑对该层做混合量化或调整量化算法

---

## 三、精度分析最佳实践

### 3.1 图片选择

- 选择**模型实际应用场景中具有代表性**的图片
- 建议挑选 1~3 张典型的测试图片即可
- 精度分析图片应**不包含在量化校准数据集中**，保证评估的客观性

### 3.2 分析后的优化路径

| 发现的问题 | 建议的优化方案 |
|-----------|--------------|
| 整体余弦相似度偏低（< 0.95） | ① 更换量化算法（如 `entropy` → `kl_divergence`）<br>② 扩充/替换量化校准数据集 |
| 仅个别层余弦相似度偏低 | 对该层/子图做**混合量化**（保留 FP16） |
| 某些层欧氏距离特别大 | 检查该层是否为激活函数层（如 Sigmoid/Softmax），<br>此类层对量化敏感，建议混合量化 |
| QNN 中某些层 Name 显示为 "lost" | 该层在 Golden 和 Quant 模型间无法匹配，<br>可能是图优化过程中被融合或消除，通常可忽略 |

### 3.3 混合量化示例（QNN）

> ⚠️ QNN 的混合量化功能尚未完全接入自动化流程，目前需要手动使用 `snpe-accuracy-debugger` CLI 进行。未来版本将支持通过 `OnnxToQNN` 直接配置。


---

## 五、注意事项

1. **QNN 精度分析仅支持 Linux 环境** — 依赖 Qualcomm `snpe-accuracy-debugger` 工具，该工具仅提供 Linux x86_64 版本。
2. **RKNN 精度分析在 Windows 和 Linux 上均可使用** — 集成在 RKNN Toolkit2 中。
3. **精度分析会增加转换时间** — 需要对同一输入在未量化和量化模型上分别做一次完整推理，耗时为正常转换的 2~3 倍。
4. **如果未设置 `accuracy_analysis_picture_list` 或设为 `None`** — 精度分析将被跳过，不影响正常转换流程。
5. **QNN 精度分析结果同时保存在 `utilities/tmp/accuracy_analysis_summary.png`** — 建议每次分析后保存此图表以便对比不同量化配置的效果。
