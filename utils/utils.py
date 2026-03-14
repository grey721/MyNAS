import hashlib
import numpy as np
import json
import os

from template.template import Template
import matplotlib.pyplot as plt
import pandas as pd


def plot_population(p1_obj, p2_obj, save_dir, generation):
    """
    可视化两个种群在三个二维平面上的投影，不使用PCA。
    输出三张图片：
        1. (目标1, 目标2)
        2. (目标1, 目标3)
        3. (目标2, 目标3)

    参数:
        p1_obj: ndarray, shape=(n1, m), 种群1目标值矩阵（不含fitness列）
        p2_obj: ndarray, shape=(n2, m), 种群2目标值矩阵（不含fitness列）
        save_dir: str, 保存路径
        generation: int, 当前代数，用于命名文件
    """
    os.makedirs(save_dir, exist_ok=True)

    p1_obj = p1_obj[:, 1:]
    p2_obj = p2_obj[:, 1:]

    targets = ['Error', 'Parameters', 'FLOPs']
    projections = [(1, 0), (2, 0), (2, 1)]

    for idx, (x_idx, y_idx) in enumerate(projections, 1):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(p1_obj[:, x_idx], p1_obj[:, y_idx], c='blue', label='P1', alpha=0.7)
        ax.scatter(p2_obj[:, x_idx], p2_obj[:, y_idx], c='red', label='P2', alpha=0.7)
        ax.set_xlabel(targets[x_idx])
        ax.set_ylabel(targets[y_idx])
        ax.set_title(f'Generation {generation} - Projection {targets[x_idx]} vs {targets[y_idx]}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gen{generation}_proj{x_idx}{y_idx}.png'))
        plt.close(fig)


def save_population_info(population, fitness_matrix, output_dir="results", filename="population", generation=None):
    """
    保存种群记录：包含优化目标和结构hash。
    """
    os.makedirs(output_dir, exist_ok=True)
    if generation is not None:
        filename = f'{filename}_{generation:>03d}'
    pop_size = population.shape[0]

    records = []
    for i in range(pop_size):
        # 结构压缩
        arch_str = json.dumps(population[i].tolist(), separators=(',', ':'))
        arch_hash = hashlib.md5(arch_str.encode()).hexdigest()[:10]  # 10位hash

        fitness, err, params, flops = fitness_matrix[i]
        records.append({
            "id": i,
            "Error": err,
            "Params(M)": round(params / 1e6, 2),
            "FLOPs(M)": round(flops / 1e6, 2),
            "Fitness": fitness,
            "ArchHash": arch_hash
        })

    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, f'{filename}.csv')
    df.to_csv(output_path, index=False)

    with open(os.path.join(output_dir, f"{filename}_arch.json"), "w", encoding="utf-8") as f:
        json.dump({r["ArchHash"]: population[i].tolist() for i, r in enumerate(records)}, f, indent=2, )


class SaveFile:
    def __init__(self):
        self.architecture = None
        self.dataset = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_architecture_with_individual(self, individual):
        self.architecture = individual

    def set_architecture_with_hash(self, net_hash, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            individuals = json.load(f)
        if net_hash in individuals:
            self.architecture = np.array(individuals[net_hash], dtype=np.float32)
        else:
            raise KeyError(f"哈希键 '{net_hash}' 不存在于数据中")

    def generate_py(self, file_name, project_name=None):
        if self.architecture is not None and self.dataset is not None:
            template = Template(self.dataset)
            template.save_code(self.architecture, file_name=file_name, project_name=project_name)
        else:
            print('请设置参数')
