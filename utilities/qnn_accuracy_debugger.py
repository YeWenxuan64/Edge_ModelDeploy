import os
import shutil
from pathlib import Path
from typing import Callable
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
import cv2
from matplotlib import axes
import matplotlib.pyplot as plt




class SnpeAccuracyDebugger:
    def __init__(self, tmp_dir:str, onnx_path:str, golden_dlc_path:str, quant_dlc_path:str, debugger_picture_list:list[str]):
        self.tmp_dir = Path(tmp_dir).resolve()
        self.onnx_path = Path(onnx_path).resolve()
        self.golden_dlc_path = Path(golden_dlc_path).resolve()
        self.quant_dlc_path = Path(quant_dlc_path).resolve()
        self.debugger_picture_list = [Path(p).resolve() for p in debugger_picture_list]

        self.tmp_working_dir = self.tmp_dir / 'accuracy_analysis'
        self.tmp_working_dir.mkdir(exist_ok=True)

        self.working_dir = Path.home() / 'accuracy_analysis'
        self.working_dir.mkdir(exist_ok=True)

        self.golden_dir = self.working_dir / 'golden_dir'
        self.golden_dir.mkdir(exist_ok=True)

        self.quant_dir = self.working_dir / 'quant_dir'
        self.quant_dir.mkdir(exist_ok=True)

    def get_parent_args(self, onnx_info:dict, command_runnr:Callable[[str], int]):
        self.onnx_info = onnx_info
        self.command_runnr = command_runnr

    @staticmethod
    def find_latest_subdir(base_dir: Path) -> Path:
        """在 base_dir 下找到最新的子文件夹（按文件夹名排序）"""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"base directory not found: {base_dir}")

        subdirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {base_dir}")

        latest = subdirs[0]
        print(f"Latest subdirectory: {latest}")
        return latest

    def prepare_input_data(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]], set_input_order:str='nchw'):
        analysis_input_dir = self.tmp_working_dir / f'analysis_inputs'
        analysis_input_dir.mkdir(exist_ok=True)

        input_infos:list[dict[str, str|list[int]]] = self.onnx_info['inputs']
        output_infos:list[dict[str, str|list[int]]] = self.onnx_info['outputs']

        input_tensor_path_list:list[str] = []
        input_arg_list:list[str] = []
        output_arg_list:list[str] = []

        for i, input_info in enumerate(input_infos):
            mean_value = mean_rgb[i % len(mean_rgb)]
            std_value = std_rgb[i % len(std_rgb)]
            image_path = self.debugger_picture_list[i % len(self.debugger_picture_list)]

            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_name:str = input_info["name"]
            input_shape:list[int] = input_info["shape"] # nchw
            channels, height, width = input_shape[1], input_shape[2], input_shape[3]


            # 将图片缩放到目标空间尺寸
            image_resized = cv2.resize(image_rgb, (width, height))  # [H, W, 3]
            image_float = image_resized.astype(np.float32)

            mean_arr = np.array(mean_value, dtype=np.float32)
            std_arr = np.array(std_value, dtype=np.float32)
            image_float = (image_float - mean_arr) / std_arr

            # 处理通道数不匹配的情况（如 [1,96,16,16] NCHW）
            if channels != 3:
                image_float = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
                image_float = np.tile(image_float, (1, 1, channels))[:, :, :channels] # 裁剪到精确通道数

            image_float = np.expand_dims(image_float, axis=0)  # [1, H, W, C]

            if set_input_order == 'nchw':
                # [1, H, W, C] -> [1, C, H, W]
                image_float = np.transpose(image_float, (0, 3, 1, 2))


            input_tensor_path = str(analysis_input_dir / f'{input_name}.raw')
            input_shape_str = str(image_float.shape).replace('(', '').replace(')', '').replace(' ', '')

            image_float.tofile(input_tensor_path)
            print(image_float.shape, image_float.max(), image_float.min())

            input_tensor_path_list.append(input_tensor_path)
            input_arg_list.append(f'--input_tensor "{input_name}" {input_shape_str} {input_tensor_path}')
            

        for output_info in output_infos:
            output_name:str = output_info["name"]
            output_arg_list.append(f'--output_tensor "{output_name}"')


        analysis_input_list = str(self.tmp_working_dir / f"analysis_input_list.txt")
        with open(analysis_input_list, 'w') as f:
            f.write(' '.join(input_tensor_path_list))

        print(f"Accuracy analysis input list: {analysis_input_list}")
        print(f"Accuracy analysis input tensor args: {input_arg_list}")
        print(f"Accuracy analysis output tensor args: {output_arg_list}")
        return analysis_input_list, input_arg_list, output_arg_list


    def qnn_infer(self, working_dir:str, dlc_model:str, analysis_input_list:list, input_arg_list:list, output_arg_list:list) -> int:
        qnn_sdk_root = os.environ.get('QNN_SDK_ROOT')

        input_tensor_args = ' '.join(input_arg_list)
        output_tensor_args = ' '.join(output_arg_list)

        command = f"snpe-accuracy-debugger --inference_engine"
        command += f" --working_dir {str(working_dir)}"
        command += f" --engine_path {qnn_sdk_root}"
        command += f" --architecture x86_64-linux-clang"
        command += f" --framework onnx"
        command += f" --runtime cpu"
        command += f" --stage converted"

        command += f" --model_path {str(self.onnx_path)}"
        command += f" --static_model {str(dlc_model)}"
        command += f" --input_list {analysis_input_list}"
        command += f" {input_tensor_args}"
        command += f" {output_tensor_args}"

        #command += f" --verbose"

        ret = self.command_runnr(command)
        return ret
    
    def analysis_results(self) -> int:
        framework_runner_dir = self.find_latest_subdir(self.golden_dir / 'inference_engine')
        inference_engine_dir = self.find_latest_subdir(self.quant_dir / 'inference_engine')

        command = f"snpe-accuracy-debugger --verification"
        command += f" --working_dir {str(self.working_dir)}"

        command += f" --default_verifier CosineSimilarity"
        command += f" --default_verifier MSE"
        command += f" --golden_output_reference_directory {framework_runner_dir}"
        command += f" --inference_results {inference_engine_dir}"
        command += f" --tensor_mapping {inference_engine_dir}/tensor_mapping.json"
        command += f" --graph_struct {inference_engine_dir}/model_graph_struct.json"

        command += f" --verbose"

        ret = self.command_runnr(command)
        return ret

    def show_accuracy_analysis(self):
        golden_infer_src = self.find_latest_subdir(self.golden_dir / "inference_engine")
        quant_infer_src = self.find_latest_subdir(self.quant_dir / "inference_engine")
        verification_src = self.find_latest_subdir(self.working_dir / 'verification')

        golden_dst = self.tmp_working_dir / 'golden_dir' / "inference_engine" / "latest"
        quant_dst = self.tmp_working_dir / 'quant_dir' / "inference_engine" / "latest"
        verification_dst = self.tmp_working_dir / "verification"

        # 多线程并行 copytree
        def do_copy(src: Path, dst: Path):
            print(f"Copying inference results from {src} to {dst}")
            shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True, ignore_dangling_symlinks=True)

        copy_pairs = [(golden_infer_src, golden_dst), (quant_infer_src, quant_dst), (verification_src, verification_dst)]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures:list[Future] = []
            for src, dst in copy_pairs:
                dst.parent.mkdir(parents=True, exist_ok=True) # 预创建目标父目录
                futures.append(executor.submit(do_copy, src, dst))

            for f in futures:
                f.result()

        shutil.rmtree(self.working_dir)



        # 1. 找到最新的 summary.csv
        file_path = verification_dst / 'summary.csv'
        print(f"Loading summary from: {file_path}")
        df = pd.read_csv(file_path)

        # 提取关键列
        sizes = df['Size'].values
        mses = df['mse'].values.astype(np.float64)
        cossim = df["cosinesimilarity"].values.astype(np.float32)
        n_elemments = len(cossim)

        try:
            names = df['Target Name'].values
        except KeyError:
            try:
                names = df['Name'].values
            except KeyError:
                names = None

        layer_index = range(n_elemments)

        # 2. 核心计算
        # 计算每一层的误差平方和 (SSE)
        squared_errors = mses * sizes

        # 逐层欧氏距离
        layer_euc = np.sqrt(squared_errors)

        # 数据清洗：避免在对数坐标下 log(0) 报错，将绝对的 0 值替换为极小值
        layer_euc[layer_euc == 0] = 1e-10


        # 3. 可视化绘图
        fig, (ax_euc, ax_cos) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        ax_euc:axes.Axes
        ax_cos:axes.Axes

        # 图1：欧氏距离（逐层柱状）+ MSE（右侧纵轴折线）
        ax_euc.set_yscale('linear')
        ax_euc.bar(layer_index, layer_euc, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.7, label='Euc Dist Per Layer')
        ax_euc.set_title('Euclidean Distance & MSE (Per Layer)', fontsize=14, fontweight='bold')
        ax_euc.set_ylabel('Euclidean Distance', fontsize=12)
        ax_euc.grid(True, which="both", ls="--", alpha=0.5)

        # 右侧纵轴：MSE（量级可能千分之一~个位，与欧氏距离的几十上百分离显示）
        ax_mse = ax_euc.twinx()
        ax_mse.plot(layer_index, mses, color='blue', marker='.', linestyle='-', linewidth=1, markersize=2, label='MSE Per Layer')
        ax_mse.set_ylabel('MSE', fontsize=12)

        # 合并两个轴的图例
        lines_euc, labels_euc = ax_euc.get_legend_handles_labels()
        lines_mse, labels_mse = ax_mse.get_legend_handles_labels()
        ax_euc.legend(lines_euc + lines_mse, labels_euc + labels_mse, loc='upper left')

        # 图2：余弦相似度（逐层折线 + 累计最小值叠加）
        ax_cos.set_yscale('linear')
        ax_cos.plot(layer_index, cossim, color='green', marker='.', linestyle='-', linewidth=1, markersize=2, label='Per Layer')
        ax_cos.set_title('Cosine Similarity (Per Layer)', fontsize=14, fontweight='bold')
        ax_cos.set_ylabel('Cosine Similarity', fontsize=12)
        ax_cos.set_xticks(layer_index)
        if names is not None:
            ax_cos.set_xticklabels(names, rotation=-45, ha='left', fontsize=10)
        
        ax_cos.axhline(0.99, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Warning Threshold (0.99)')
        ax_cos.legend()
        ax_cos.grid(True, which="both", ls="--", alpha=0.5)

        # 消除 x 轴两端默认的 5% 空白边距（留半个柱宽避免首尾柱被裁切）
        ax_cos.set_xlim(-0.5, len(layer_euc) - 0.5)

        # 调整布局、保存并显示
        plt.tight_layout()
        save_path = self.tmp_dir / 'accuracy_analysis_summary.png'
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.show()

    def accuracy_analysis(self, mean_rgb:list[list[int|float,]]=[[0, 0, 0]], std_rgb:list[list[int|float,]]=[[1, 1, 1]], set_input_order:str='nhwc') -> int:
        analysis_input_list, input_arg_list, output_arg_list = self.prepare_input_data(mean_rgb, std_rgb, set_input_order)
        ret_golden, ret_quant = 1, 1
        
        # 并行执行 Golden（未量化）和 Quantized（量化）推理，两者互不依赖
        with ThreadPoolExecutor(max_workers=2) as executor:
            golden_future = executor.submit(
                self.qnn_infer, self.golden_dir, self.golden_dlc_path,
                analysis_input_list, input_arg_list, output_arg_list
            )
            quant_future = executor.submit(
                self.qnn_infer, self.quant_dir, self.quant_dlc_path,
                analysis_input_list, input_arg_list, output_arg_list
            )
            ret_golden = golden_future.result()
            ret_quant = quant_future.result()

        if ret_golden != 0:
            print(f"Failed to run framework runner")
            exit(1)
        
        if ret_quant != 0:
            print(f"Failed to run inference engine")
            exit(1)

        ret = self.analysis_results()
        if ret != 0:
            print(f"Failed to run verification")
            exit(1)

        #self.custom_accuracy_analysis()

        self.show_accuracy_analysis()

        return ret

    def custom_accuracy_analysis(self):
        framework_runner_dir = self.find_latest_subdir(self.golden_dir / 'inference_engine')
        inference_engine_dir = self.find_latest_subdir(self.quant_dir / 'inference_engine')
        latest_ver_dir = self.find_latest_subdir(self.working_dir / "verification")

        framework_output_dir = framework_runner_dir / "output" / "Result_0"
        inference_output_dir = inference_engine_dir / "output" / "Result_0"

        file_path = str(latest_ver_dir / 'summary.csv')
        df = pd.read_csv(file_path)

        # 提取关键列
        try:
            names = df['Target Name'].values
        except KeyError:
            names = df['Name'].values


        # 1. 收集两个目录下的所有 .raw 文件
        golden_raw_files: dict[str, Path] = {}
        for f in framework_output_dir.rglob("*.raw"):
            golden_raw_files[f.name] = f

        infer_raw_files: dict[str, Path] = {}
        for f in inference_output_dir.rglob("*.raw"):
            infer_raw_files[f.name] = f

        if not golden_raw_files:
            print(f"No .raw files found in {framework_output_dir}")
            return
        if not infer_raw_files:
            print(f"No .raw files found in {inference_output_dir}")
            return

        # 2. 使用 SequenceMatcher 匹配相似文件名
        golden_names = list(golden_raw_files.keys())
        infer_names = list(infer_raw_files.keys())

        matched_pairs: list[tuple[str, str, float]] = []  # (golden_name, infer_name, similarity)
        unmatched_infer = set(infer_names)

        for g_name in golden_names:
            best_score = 0.0
            best_match = None
            for i_name in unmatched_infer:
                score = SequenceMatcher(None, g_name, i_name).ratio()
                if score > best_score:
                    best_score = score
                    best_match = i_name
            if best_match is not None:
                matched_pairs.append((g_name, best_match, best_score))
                unmatched_infer.discard(best_match)
            else:
                matched_pairs.append((g_name, None, 0.0))

        # 未匹配的推理文件
        for i_name in unmatched_infer:
            matched_pairs.append((None, i_name, 0.0))

        # 3. 加载 raw 文件并计算余弦距离
        NAME_SIM_THRESHOLD = 0.8

        results: list[dict] = []
        lost_pairs: list[dict] = []  # 名称相似度<0.8 或 尺寸不一致的配对

        for g_name, i_name, name_sim in matched_pairs:
            if g_name is None:
                lost_pairs.append({"golden": "<MISSING>", "infer": i_name, "reason": "no golden match"})
                continue
            if i_name is None:
                lost_pairs.append({"golden": g_name, "infer": "<MISSING>", "reason": "no infer match"})
                continue

            # 名称相似度不足，标记为丢失
            if name_sim < NAME_SIM_THRESHOLD:
                lost_pairs.append({
                    "golden": g_name, "infer": i_name,
                    "name_similarity": name_sim,
                    "reason": f"name similarity {name_sim:.4f} < {NAME_SIM_THRESHOLD}"
                })
                continue

            try:
                g_path = golden_raw_files[g_name]
                i_path = infer_raw_files[i_name]

                # 读取 raw 文件为 float32 并展平
                g_data = np.fromfile(str(g_path), dtype=np.float32).flatten()
                i_data = np.fromfile(str(i_path), dtype=np.float32).flatten()

                # 尺寸不一致，标记为丢失
                if len(g_data) != len(i_data):
                    lost_pairs.append({
                        "golden": g_name, "infer": i_name,
                        "name_similarity": name_sim,
                        "reason": f"size mismatch: golden={len(g_data)}, infer={len(i_data)}"
                    })
                    continue

                # 计算余弦相似度
                dot_product = np.dot(g_data, i_data)
                norm_g = np.linalg.norm(g_data)
                norm_i = np.linalg.norm(i_data)

                if norm_g == 0 or norm_i == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = dot_product / (norm_g * norm_i)

                cos_dist = 1.0 - cos_sim

                # 欧几里得距离: ||g - i||₂
                diff = g_data - i_data
                euc_dist = float(np.linalg.norm(diff))

                results.append({
                    "golden": g_name,
                    "infer": i_name,
                    "name_similarity": name_sim,
                    "cosine_similarity": float(cos_sim),
                    "cosine_distance": float(cos_dist),
                    "euclidean_distance": euc_dist,
                })

            except Exception as e:
                lost_pairs.append({
                    "golden": g_name, "infer": i_name,
                    "name_similarity": name_sim,
                    "reason": f"error: {e}"
                })

        # 4. 按 CSV 中的 'Target Name'/'Name' 列顺序排序
        # names 已在前面从 summary.csv 中提取
        csv_name_to_index = {name: idx for idx, name in enumerate(names)}

        for r in results:
            g_name = r["golden"]
            best_idx = len(names)  # 默认放在末尾

            best_score = 0.0
            for csv_name, csv_idx in csv_name_to_index.items():
                # 匹配 golden 文件名与 CSV 中的名称
                score_g = SequenceMatcher(None, g_name, csv_name).ratio()
                if score_g > best_score:
                    best_score = score_g
                    best_idx = csv_idx

            # 只有相似度足够高（>0.6）才采纳 CSV 顺序，否则放末尾
            if best_score < 0.6:
                best_idx = len(names)

            r["mapping_index"] = best_idx

        # 先按 CSV 名称顺序，再按欧几里得距离（同组内降序）
        results.sort(key=lambda r: (r["mapping_index"], -r["euclidean_distance"]))
        if lost_pairs:
            print(f"\n{'='*100}")
            print(f"LOST PAIRS ({len(lost_pairs)} pairs) — not included in cosine distance stats:")
            print(f"{'='*100}")
            print(f"{'Golden File':<50} {'Infer File':<50} {'Reason'}")
            print(f"{'-'*100}")
            for lp in lost_pairs:
                g = lp["golden"]
                i = lp["infer"]
                reason = lp.get("reason", "-")
                print(f"{g:<50} {i:<50} {reason}")

        # 6. matplotlib 可视化（按 CSV 名称顺序排列）
        if results:
            n = len(results)
            euc_vals = np.array([r["euclidean_distance"] for r in results])
            cos_sim_vals = np.array([r["cosine_similarity"] for r in results])

            def shorten(name: str, max_len: int = 50) -> str:
                name = name.replace('.raw', '')
                if len(name) > max_len:
                    return name[:max_len - 3] + '...'
                return name

            names = [shorten(r["golden"]) for r in results]
            layer_index = np.arange(n)

            # 数据清洗：避免 log(0)，将 0 替换为极小值
            euc_vals[euc_vals == 0] = 1e-10

            fig, (ax_euc, ax_cos) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

            # ---- 图1：欧几里得距离（逐层柱状） ----
            ax_euc.bar(layer_index, euc_vals, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.7, label='Per Layer')
            ax_euc.set_title(f'Euclidean Distance (ordered by CSV Target Name, {n} valid / {len(lost_pairs)} lost)', fontsize=14, fontweight='bold')
            ax_euc.set_ylabel('Euclidean Distance', fontsize=12)
            ax_euc.legend()
            ax_euc.grid(True, which="both", ls="--", alpha=0.5)

            # ---- 图2：余弦相似度（逐层折线） ----
            ax_cos.plot(layer_index, cos_sim_vals, color='green', marker='.', linestyle='-', linewidth=1.5, markersize=4, label='Per Layer')
            ax_cos.set_title('Cosine Similarity (Per Tensor Pair)', fontsize=14, fontweight='bold')
            ax_cos.set_ylabel('Cosine Similarity', fontsize=12)
            ax_cos.set_xticks(layer_index)
            ax_cos.set_xticklabels(names, rotation=90, fontsize=6)
            ax_cos.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='base')
            ax_cos.axhline(0.99, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Warning Threshold (0.99)')
            ax_cos.legend()
            ax_cos.grid(True, which="both", ls="--", alpha=0.5)

            plt.tight_layout()
            plt.show()

        # 7. 终端汇总统计
        if results:
            euc_vals = [r["euclidean_distance"] for r in results]
            cos_dists = [r["cosine_distance"] for r in results]
            cos_sims = [r["cosine_similarity"] for r in results]
            print(f"\n{'='*80}")
            print(f"Summary: {len(results)} valid + {len(lost_pairs)} lost = {len(results) + len(lost_pairs)} total")
            print(f"  Euclidean Distance - min: {min(euc_vals):.6f}, max: {max(euc_vals):.6f}, mean: {np.mean(euc_vals):.6f}")
            print(f"  Cosine Distance    - min: {min(cos_dists):.6f}, max: {max(cos_dists):.6f}, mean: {np.mean(cos_dists):.6f}")
            print(f"  Cosine Similarity  - min: {min(cos_sims):.6f}, max: {max(cos_sims):.6f}, mean: {np.mean(cos_sims):.6f}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Summary: 0 valid pairs, {len(lost_pairs)} lost — no cosine distance stats available.")
            print(f"{'='*80}\n")

        return results


