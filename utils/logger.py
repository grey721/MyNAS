import json
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from utils.utils import plot_population, save_population_info


class Logger:
    def __init__(self, exp_name="Experiment", base_dir="logs", log_to_console=True, debug=False):
        # 实验目录
        if debug:
            self.project_name = exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.project_name = f"{exp_name}_{timestamp}"

        self.exp_dir = os.path.join(base_dir, self.project_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.log_file = os.path.join(self.exp_dir, f"{exp_name}.log")

        # logging 初始化
        self.logger = logging.getLogger(self.project_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(levelname)-8s   %(asctime)s | %(message)s", "%H:%M:%S")

        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        if log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    # ------------------------------------------------------------------
    # 基础日志接口
    # ------------------------------------------------------------------

    def info(self, msg):    self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg):   self.logger.error(msg)
    def debug(self, msg):   self.logger.debug(msg)

    def save_config(self, args):
        path = os.path.join(self.exp_dir, 'config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
        self.info(f"Config saved → {path}")

    # ------------------------------------------------------------------
    # NAS 种群可视化（原有功能保留）
    # ------------------------------------------------------------------

    def plot_pop(self, p1, p2, generation):
        out = os.path.join(self.exp_dir, 'pic')
        plot_population(p1, p2, generation=generation, save_dir=out)
        self.info(f'Population plot saved → {out}')

    def save_population(self, population, fitness_matrix, filename, generation=None):
        out = os.path.join(self.exp_dir, 'population')
        save_population_info(population=population, fitness_matrix=fitness_matrix,
                             filename=filename, output_dir=out, generation=generation)
        self.info(f'Population info saved → {out}')

    # ------------------------------------------------------------------
    # 训练曲线可视化
    # ------------------------------------------------------------------

    def plot_training(self, history: dict, title: str = 'Training Curves', filename: str = 'training_curves.png'):
        """
        根据 history 字典绘制训练曲线并保存。

        history 格式：
        {
            'train_loss': [...], 'test_loss': [...],
            'train_acc':  [...], 'test_acc':  [...],
            'lr':         [...]
        }
        """
        out_dir = os.path.join(self.exp_dir, 'pic')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)

        epochs = range(1, len(history['train_loss']) + 1)

        fig = plt.figure(figsize=(15, 4))
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # --- Loss ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue',  linewidth=1.5)
        ax1.plot(epochs, history['test_loss'],  label='Test',  color='tomato',     linewidth=1.5)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        self._mark_best(ax1, epochs, history['test_loss'], mode='min', color='tomato')

        # --- Accuracy ---
        ax2 = fig.add_subplot(gs[1])
        train_acc_pct = [a * 100 for a in history['train_acc']]
        test_acc_pct  = [a * 100 for a in history['test_acc']]
        ax2.plot(epochs, train_acc_pct, label='Train', color='steelblue', linewidth=1.5)
        ax2.plot(epochs, test_acc_pct,  label='Test',  color='tomato',    linewidth=1.5)
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Acc (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        best_acc = self._mark_best(ax2, epochs, test_acc_pct, mode='max', color='tomato')
        if best_acc is not None:
            ax2.set_title(f'Accuracy  (best test: {best_acc:.2f}%)')

        # --- Learning Rate ---
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(epochs, history['lr'], color='mediumseagreen', linewidth=1.5)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.info(f'Training curves saved → {save_path}')

    def plot_training_realtime(self, history: dict, epoch: int,
                               total_epochs: int, interval: int = 10,
                               title: str = 'Training Curves'):
        """
        每隔 interval 个 epoch 自动保存一次曲线（实时更新用）。
        最终图命名为 training_curves.png，中间图命名为 training_curves_eXXX.png。
        """
        if epoch % interval == 0 or epoch == total_epochs:
            filename = ('training_curves.png' if epoch == total_epochs
                        else f'training_curves_e{epoch:04d}.png')
            self.plot_training(history, title=title, filename=filename)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_best(ax, epochs, values, mode='min', color='tomato', markersize=6):
        """在最优点打星标并标注数值，返回最优值。"""
        if not values:
            return None
        best_idx = int(np.argmin(values) if mode == 'min' else np.argmax(values))
        best_val = values[best_idx]
        best_ep  = list(epochs)[best_idx]
        ax.scatter(best_ep, best_val, color=color, zorder=5,
                   marker='*', s=markersize ** 2)
        ax.annotate(f'{best_val:.3f}',
                    xy=(best_ep, best_val),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=7.5, color=color)
        return best_val
