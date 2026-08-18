import re
import sys
import os
from pathlib import Path
import numpy as np
from rknn.api import RKNN


current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from utils import temporary_chdir, clean_files_or_dirs, read_dataset_txt_to_list




class OnnxToRKNN:
    def __init__(self, model_path:str, rknn_model_path:str, dataset_path:str|None=None, target_platform:str='rk3588'):
        """
        Initialize the ONNX to RKNN converter.

        Args:
            model_path (str): Path to the input ONNX model file that needs to be converted.

            rknn_model_path (str): Path where the converted RKNN model file will be saved.

            dataset_path (str | None): Path to a text file containing paths to dataset images for quantization. 
                - The text file should contain one image path per line for single-input models, 
                or multiple image paths separated by spaces for multi-input models.
                - Default is None. no quantization will be performed.

            target_platform (str): Target platform for the converted model. 
                - Supported platforms are 'rk3588', 'rk3576', 'rk3566'.
                - Defaults to 'rk3588'.
        """
        
        current_dir = Path(__file__).parent.resolve() # 获取当前文件所在目录的绝对路径
        self.tmp_dir = current_dir / 'tmp' # 构建tmp目录的绝对路径

        self.model_path = Path(model_path).resolve()
        self.rknn_model_path = Path(rknn_model_path).resolve()
		
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).resolve()
        else:
            self.dataset_path = None

        self.target_platform = target_platform
        if self.target_platform not in ['rk3588', 'rk3576', 'rk3566']:
            raise ValueError("target_platform must be 'rk3588' or 'rk3576' or 'rk3566'")

        # self.extra_optimize()
        self.quantized_algorithm = 'normal'
        self.compress_weight = False
        self.model_pruning = False
        self.flash_attantion = False

        # self.do_hybrid_quantization()
        self.custom_hybrid = None
        
        #self.set_do_accuracy_analysis()
        self.accuracy_analysis_picture_list = None

        self.file_or_dir_to_clean = ["check0_base_optimize.onnx", "check1_fold_constant.onnx", "check2_correct_ops.onnx", "check3_fuse_ops.onnx"]

    def extra_optimize(self, quantized_algorithm:str='normal', compress_weight:bool=False, model_pruning:bool=False, flash_attantion:bool=False):
        """
        Args:
            quantized_algorithm (str): The quantization algorithm to use. 
                - Options: 'normal' for min-max quantization, 'kl_divergence' for KL divergence-based or 'mmse' for minimum mean square error quantization.
                - Default is 'normal'.

            compress_weight (bool): Whether to compress model weights to reduce memory usage. 
                - Default is False.

            model_pruning (bool): Whether to apply model pruning to remove less important parameters. 
                - Default is False.

            flash_attantion (bool): Whether to use flash attention mechanism for faster attention computation. 
                - Default is False.
        """

        if quantized_algorithm not in ['normal', 'kl_divergence', 'mmse']:
            raise ValueError("quantized_algorithm must be 'normal' or 'kl_divergence' or 'mmse'")
        
        self.quantized_algorithm = quantized_algorithm
        self.compress_weight = compress_weight
        self.model_pruning = model_pruning
        self.flash_attantion = flash_attantion

        print(f"[OnnxToRKNN] extra_optimize: quantized_algorithm={self.quantized_algorithm}, compress_weight={self.compress_weight}, model_pruning={self.model_pruning}, flash_attantion={self.flash_attantion}")

    def do_hybrid_quantization(self, custom_hybrid:list[list[str]]|None=None):
        """
        Args:
            custom_hybrid (list[list[str]], optional): A list of onnx node's input and output pair specifying the custom hybrid quantization settings.
                - Each inner list contains two strings representing the input name and output name of a subgraph in the ONNX model.
                - All nodes between the specified input and output will be quantized using FP16, 
                - while nodes outside these subgraphs will remain in 8-bit quantization. 
                - To apply hybrid quantization to multiple subgraphs, provide multiple pairs in the list, 
                e.g., [[input_name1, output_name1], [input_name2, output_name2]]. 
                - Defaults to None.
        """
        self.custom_hybrid = custom_hybrid

        print(f"[OnnxToRKNN] do_hybrid_quantization: custom_hybrid={self.custom_hybrid}")

    def set_do_accuracy_analysis(self, accuracy_analysis_picture_list:list[str]|None=None):
        """
        Args:
            accuracy_analysis_picture_list (list[str], optional): A list of image paths required for model accuracy analysis. 
                - Each element in the list should be a path to an image. 
                - For models with a single input, provide a single image path. 
                - For models with multiple inputs, provide multiple image paths. Example: ['/home/xxx/1.jpg', '/home/xxx/2.jpg']
                - Defaults to None.
        """
        if accuracy_analysis_picture_list is not None:
            self.accuracy_analysis_picture_list = [str(Path(path).resolve()) for path in accuracy_analysis_picture_list]
        else:
            self.accuracy_analysis_picture_list = None

        print(f"[OnnxToRKNN] set_do_accuracy_analysis: accuracy_analysis_picture_list={self.accuracy_analysis_picture_list}")


    def convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        """
        Args:
            mean_rgb (list[list[int | float,]], optional): Mean values for RGB channels normalization.
                - Each inner list contains 3 values (R, G, B) representing the mean for each channel in one input.
                - If multiple inputs are provided, For example, [[123, 116, 103], [123, 116, 103]]
                - Defaults to [[0, 0, 0]] (no mean normalization).
                
            std_rgb (list[list[int | float,]], optional): Standard deviation values for RGB channels normalization.
                - Each inner list should contain 3 values (R, G, B) representing the standard deviation for each channel in one input.
                - Similar to mean_rgb, can provide multiple lists for multiple inputs.
                - Defaults to [[1, 1, 1]] (no standard deviation normalization).
        """

        self.tmp_dir.mkdir(exist_ok=True)

        if self.dataset_path is not None: # 读取数据集文件
            dataset_path_list = read_dataset_txt_to_list(self.dataset_path)

            tmp_dataset_path = self.tmp_dir / self.dataset_path.name
            with open(tmp_dataset_path, 'w') as f:
                for paths in dataset_path_list:
                    f.write(' '.join(paths) + '\n')

            self.dataset_path = tmp_dataset_path
            self.file_or_dir_to_clean.append(self.dataset_path)

        with temporary_chdir(self.tmp_dir):
            self.self_convert(mean_rgb, std_rgb)

        if self.accuracy_analysis_picture_list is not None:
            self.plot_accuracy_analysis()

    def clean(self):
        clean_files_or_dirs([str(self.tmp_dir / name) for name in self.file_or_dir_to_clean])


    def self_convert(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]]):
        rknn = RKNN(verbose=True)

        # Pre-process config
        print('--> Config model')
        rknn.config(mean_values=mean_rgb, std_values=std_rgb, quantized_algorithm=self.quantized_algorithm, target_platform=self.target_platform, 
                    compress_weight=self.compress_weight, model_pruning=self.model_pruning, enable_flash_attention=self.flash_attantion)
        print('done')

        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(model=str(self.model_path))
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')
        
        # Build model
        print('--> Building model')
        if self.dataset_path is not None:
            if self.custom_hybrid is None:
                ret = rknn.build(do_quantization=True, dataset=self.dataset_path)
            else:
                model_name = self.model_path.stem  # 获取文件名不带扩展名
                model_input = model_name + ".model" # 表示第一步生成的模型文件
                data_input = model_name + ".data" # 表示第一步生成的配置文件
                model_quantization_cfg = model_name + ".quantization.cfg" # 表示第一步生成的量化配置文件
                self.file_or_dir_to_clean.extend([model_input, data_input, model_quantization_cfg])

                ret = rknn.hybrid_quantization_step1(dataset=self.dataset_path, proposal=False, custom_hybrid=self.custom_hybrid)
                ret = rknn.hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)  
        else:
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export rknn model')
        
        os.makedirs(self.rknn_model_path.parent, exist_ok=True)
        
        ret = rknn.export_rknn(str(self.rknn_model_path))
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        if self.accuracy_analysis_picture_list is not None:
            print(f'accuracy_analysis_picture_list: {self.accuracy_analysis_picture_list}')
            rknn.accuracy_analysis(inputs=self.accuracy_analysis_picture_list)
            self.file_or_dir_to_clean.append(str(self.tmp_dir / "snapshot"))

        # Release
        rknn.release()
        print('--> Released rknn')


    def read_error_analysis(self) -> list[dict]:
        """
        读取 RKNN 精度分析结果文件 (snapshot/error_analysis.txt)。

        Args:
            error_analysis_path (str | None): 精度分析结果文件路径。
                - None (默认): 使用 self.tmp_dir/snapshot/error_analysis.txt。

        Returns:
            list[dict]: 逐层精度数据列表，每个元素包含:
                - op_type (str): 算子类型，如 'Conv'、'LeakyRelu'
                - layer_name (str): 层名称
                - entire_cos (float | None): 累积余弦相似度（从输入累计到该层）
                - entire_euc (float | None): 累积欧氏距离
                - single_cos (float | None): 单层余弦相似度（反映该层自身量化误差）
                - single_euc (float | None): 单层欧氏距离
                - 少数层（如部分 Concat）无数值，对应字段为 None。

        Raises:
            FileNotFoundError: 当结果文件不存在时。
        """

        error_analysis_path = self.tmp_dir / 'snapshot' / 'error_analysis.txt'

        if not error_analysis_path.exists():
            raise FileNotFoundError(
                f"error_analysis.txt not found: {error_analysis_path}\n"
                "请先通过 set_do_accuracy_analysis() 启用精度分析并运行 convert() 生成结果。"
            )

        rows: list[dict] = []
        with open(error_analysis_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                # 跳过注释行、表头行和分隔线
                if line.startswith('#') or line.startswith('layer_name') or line.startswith('---'):
                    continue

                # 数据行格式: [OpType] layer_name   entire_cos | entire_euc   single_cos | single_euc
                if not line.startswith('['):
                    continue
                end_bracket = line.find(']')
                if end_bracket == -1:
                    continue
                op_type = line[1:end_bracket]
                rest = line[end_bracket + 1:].strip()

                # 匹配末尾的 4 个数值: cos | euc   cos | euc
                m = re.match(r'^(.*?)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s*$', rest)
                if m:
                    layer_name = m.group(1).strip()
                    entire_cos = float(m.group(2))
                    entire_euc = float(m.group(3))
                    single_cos = float(m.group(4))
                    single_euc = float(m.group(5))
                else:
                    # 无数值的层（如部分 Concat 层）
                    layer_name = rest
                    entire_cos = entire_euc = single_cos = single_euc = None

                rows.append({
                    'op_type': op_type,
                    'layer_name': layer_name,
                    'entire_cos': entire_cos,
                    'entire_euc': entire_euc,
                    'single_cos': single_cos,
                    'single_euc': single_euc,
                })

        print(f"Loaded {len(rows)} layers from {error_analysis_path}")
        return rows

    def plot_accuracy_analysis(self):
        """
        读取并可视化 RKNN 精度分析结果（欧氏距离柱状图 + 余弦相似度折线图）。
        """
        from matplotlib import axes
        import matplotlib.pyplot as plt

        rows = self.read_error_analysis()
        if not rows:
            print("No data to plot.")
            return rows

        layer_names = [r['layer_name'] for r in rows]
        n = len(rows)
        x = np.arange(n)

        entire_cos = np.array([r['entire_cos'] for r in rows], dtype=float)
        entire_euc = np.array([r['entire_euc'] for r in rows], dtype=float)
        single_cos = np.array([r['single_cos'] for r in rows], dtype=float)
        single_euc = np.array([r['single_euc'] for r in rows], dtype=float)

        # 图例颜色/样式与 QNN 精度分析保持一致
        fig, (ax_euc, ax_cos) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        ax_euc:axes.Axes
        ax_cos:axes.Axes

        # 图1：欧氏距离（逐层柱状）+ 累积欧氏距离（右侧纵轴折线）
        ax_euc.set_yscale('linear')
        ax_euc.bar(x, single_euc, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.7, label='Euc Dist Per Layer')
        ax_euc.set_title('Euclidean Distance & Euc(entire) (Per Layer)', fontsize=14, fontweight='bold')
        ax_euc.set_ylabel('Euclidean Distance', fontsize=12)
        ax_euc.grid(True, which='both', ls='--', alpha=0.5)

        # 右侧纵轴：累积欧氏距离（entire，与单层欧氏距离量级接近，分离显示便于对比）
        ax_entire = ax_euc.twinx()
        ax_entire.plot(x, entire_euc, color='orange', marker='.', linestyle='-', linewidth=1.5, markersize=2, label='Euc (entire)')
        ax_entire.set_ylabel('Euc (entire)', fontsize=12)

        # 合并两个轴的图例
        lines_euc, labels_euc = ax_euc.get_legend_handles_labels()
        lines_entire, labels_entire = ax_entire.get_legend_handles_labels()
        ax_euc.legend(lines_euc + lines_entire, labels_euc + labels_entire, loc='upper left')

        # 图2：余弦相似度（逐层折线）
        ax_cos.set_yscale('linear')
        ax_cos.plot(x, single_cos, color='green', marker='.', linestyle='-', linewidth=1, markersize=2, label='Cosine (single)')
        ax_cos.plot(x, entire_cos, color='orange', marker='.', linestyle='-', linewidth=1.5, markersize=2, label='Cosine (entire)')
        ax_cos.set_title('Cosine Similarity (Per Layer)', fontsize=14, fontweight='bold')
        ax_cos.set_ylabel('Cosine Similarity', fontsize=12)
        ax_cos.set_xticks(range(n))
        ax_cos.set_xticklabels(layer_names, rotation=-45, ha='left', fontsize=10)

        ax_cos.axhline(0.99, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Warning Threshold (0.99)')
        ax_cos.legend(loc='lower left')
        ax_cos.grid(True, which='both', ls='--', alpha=0.5)

        # 消除 x 轴两端默认的 5% 空白边距（留半个柱宽避免首尾柱被裁切）
        ax_cos.set_xlim(-0.5, n - 0.5)


        plt.tight_layout()
        save_path = self.tmp_dir / 'rknn_accuracy_analysis_summary.png'
        self.file_or_dir_to_clean.append(save_path)

        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.show()



