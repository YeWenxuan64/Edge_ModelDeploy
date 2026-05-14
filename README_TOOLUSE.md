# 工具链使用指南

## 使用转换工具转换自己的模型
仿造示例的子模块，新建文件夹转换工程文件夹，并按照以特定结构放置文件和填写转换脚本
```
Focus-Finder_ModelDeploy/
├── utilities/
│   └── ...
├── datasets/                       
│   └── ...                         
├── Your_ModelDeploy/                         # 新建该模型的转换工程文件夹
│   ├── models_convert/                
|   |   ├── original/                         # 原始模型目录
|   |   ├── onnx/              
|   |   └── ...
|   ├── Your_Model_originalframework2onnx.py  # PyTorch → ONNX 转换脚本
|   ├── Your_Model_onnx2rknn.py               # ONNX → RKNN 转换脚本（如果需要）
|   └── Your_Model_onnx2qnn.py                # ONNX → QNN 转换脚本（如果需要）
└── ...
```

### `OnnxToRKNN` — ONNX 转 Rockchip RKNN

```python
from utilities.onnx_to_rknn import OnnxToRKNN

converter = OnnxToRKNN(
    model_path='path/to/model.onnx',
    rknn_model_path='path/to/output.rknn',
    dataset_path='path/to/dataset.txt',   # 量化校准数据集（可选）
    target_platform='rk3588'               # 或 'rk3576'
)

# 可选：高级优化配置
converter.extra_optimize(
    quantized_algorithm='kl_divergence',   # 'normal' / 'kl_divergence' / 'mmse'
    flash_attantion=True,
    compress_weight=False,
    model_pruning=False
)

# 可选：混合量化（部分子图 FP16，其余 INT8）
converter.do_hybrid_quantization(
    custom_hybrid=[['input_node', 'output_node']]
)

# 可选：精度分析
converter.set_do_accuracy_analysis(
    accuracy_analysis_picture_list=['img1.jpg', 'img2.jpg']
)

# 执行转换
converter.convert(
    mean_rgb=[[123.675, 116.28, 103.53]],
    std_rgb=[[58.395, 57.12, 57.375]]
)
```



### `OnnxToQNN` — ONNX 转 Qualcomm QNN

```python
from utilities.onnx_to_qnn import OnnxToQNN

converter = OnnxToQNN(
    model_path='path/to/model.onnx',
    qnn_model_path='path/to/output.bin',
    dataset_path='path/to/dataset.txt'   # 量化校准数据集（可选）
)

# 可选：量化配置
converter.set_quantization_method(
    param_quant_method='entropy',    # 'min-max' / 'sqnr' / 'percentile' / 'mse' / 'entropy'
    act_quant_method='entropy',
    bitwidth='w8a8',                 # 'w4a8' / 'w4a16' / 'w8a8' / 'w8a16' / 'w16a16'
    bias_bitwidth=8                  # 8 或 32
)

# 可选：使用自定义校准数据（.raw 格式）
converter.use_custom_alibration_data(
    custom_alibration_data_path='path/to/calibration_data.txt'
)

# 执行转换
converter.convert(
    mean_rgb=[[123.675, 116.28, 103.53]],
    std_rgb=[[58.395, 57.12, 57.375]],
    set_input_order='nhwc'           # 或 'nchw'
)
```

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

### `GenYoloDetedDataset` — 量化校准数据集生成

```python
from utilities.yolo_det_dataset_gen import GenYoloDetedDataset

generator = GenYoloDetedDataset(
    dataset_path='datasets/datasets.txt',
    output_dir_name='cropped_images'
)

# 生成裁剪后的数据集（基于 YOLO 检测结果自动裁剪感兴趣区域）
dataset_list_path = generator.gerenate(swap_image_pair=True)

# 清理临时文件
generator.clean()
```

**数据集文件格式：** 每行包含一对或多张图片路径（空格分隔），对应模型的多输入。