from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import matplotlib

matplotlib.use('Agg')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.saver import plot_population, save_population_info


class Logger:
    """
    Experiment logger.

    Parameters
    ----------
    name      : Experiment name prefix, e.g. 'Train_Best_architecture'.
    log_dir   : Root directory for logs.  Default: 'logs'.
    console   : Whether to also print to stdout.
    overwrite : If True, the run directory is not timestamped
                (useful for debugging or resuming a run).
    """

    PLOTS_DIR      = 'plots'
    POPULATION_DIR = 'population'

    def __init__(
            self,
            name: str = 'Experiment',
            log_dir: str = 'logs',
            console: bool = True,
            overwrite: bool = False,
    ) -> None:
        if overwrite:
            self.run_name = name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_name = f'{name}_{timestamp}'

        self.exp_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        log_file = os.path.join(self.exp_dir, 'run.log')
        self._logger = logging.getLogger(self.run_name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        fmt = logging.Formatter(
            '%(levelname)-8s   %(asctime)s | %(message)s',
            datefmt='%H:%M:%S',
        )
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

    # ------------------------------------------------------------------
    # Basic logging interface
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self, args) -> None:
        """Serialize an argparse.Namespace or dict to config.json."""
        path = os.path.join(self.exp_dir, 'config.json')
        data = vars(args) if hasattr(args, '__dict__') else dict(args)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.info(f'Config saved -> {path}')

    # ------------------------------------------------------------------
    # NAS population visualisation & data persistence
    # ------------------------------------------------------------------

    def plot_pop(
            self,
            p1_fitness: np.ndarray,
            p2_fitness: np.ndarray,
            generation: int,
    ) -> None:
        out_dir = os.path.join(self.exp_dir, self.PLOTS_DIR)
        plot_population(p1_fitness, p2_fitness, save_dir=out_dir, generation=generation)
        self.info(f'Population scatter plot saved -> {out_dir}')

    def save_population(
            self,
            population: np.ndarray,
            fitness_matrix: np.ndarray,
            label: str,
            generation: int | None = None,
    ) -> None:
        out_dir = os.path.join(self.exp_dir, self.POPULATION_DIR)
        save_population_info(
            population=population,
            fitness_matrix=fitness_matrix,
            label=label,
            generation=generation,
            output_dir=out_dir,
        )
        self.info(f'Population data saved -> {out_dir}')

    # ------------------------------------------------------------------
    # Training history: data persistence
    # ------------------------------------------------------------------

    def save_history(self, history: dict) -> None:
        """
        Write complete training history to disk as CSV and JSON.

        Called once at the end of training (including interrupted / failed
        runs).  Each call overwrites the previous files, so the output is
        always a complete, consistent snapshot of whatever epochs were run.

        Output files
        ------------
        training_history.csv
            Per-epoch table for direct use in visualisation / comparison
            scripts.  Columns:

                epoch        — 1-based epoch index
                train_loss   — mean cross-entropy on the training set
                test_loss    — mean cross-entropy on the test set
                train_acc    — training accuracy  [0, 1]
                test_acc     — test accuracy      [0, 1]
                lr           — learning rate at the start of the epoch

        training_summary.json
            Scalar summary for quick inspection.  Fields:

                best_acc         — highest test_acc observed
                best_acc_epoch   — epoch at which best_acc was achieved
                best_loss        — lowest test_loss observed
                best_loss_epoch  — epoch at which best_loss was achieved
                final_train_acc  — train_acc at the last completed epoch
                final_test_acc   — test_acc  at the last completed epoch
                final_lr         — lr at the last completed epoch

        Parameters
        ----------
        history : dict
            Keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'lr'.
            All lists must have the same length (= number of completed epochs).
        """
        n_epochs = len(history['train_loss'])
        if n_epochs == 0:
            return

        train_acc = history['train_acc']
        test_acc  = history['test_acc']

        # ---- CSV --------------------------------------------------------
        df = pd.DataFrame({
            'epoch':      list(range(1, n_epochs + 1)),
            'train_loss': history['train_loss'],
            'test_loss':  history['test_loss'],
            'train_acc':  train_acc,
            'test_acc':   test_acc,
            'lr':         history['lr'],
        })
        csv_path = os.path.join(self.exp_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        self.info(f'Training history CSV saved -> {csv_path}')

        # ---- JSON summary -----------------------------------------------
        best_acc_idx  = int(np.argmax(test_acc))
        best_loss_idx = int(np.argmin(history['test_loss']))

        summary = {
            'best_acc':        round(test_acc[best_acc_idx], 6),
            'best_acc_epoch':  best_acc_idx + 1,
            'best_loss':       round(history['test_loss'][best_loss_idx], 6),
            'best_loss_epoch': best_loss_idx + 1,
            'final_train_acc': round(train_acc[-1], 6),
            'final_test_acc':  round(test_acc[-1], 6),
            'final_lr':        history['lr'][-1],
        }
        summary_path = os.path.join(self.exp_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        self.info(f'Training summary JSON saved -> {summary_path}')

    # ------------------------------------------------------------------
    # Training curve visualisation
    # ------------------------------------------------------------------

    def plot_training(
            self,
            history: dict,
            title: str = 'Training Curves',
            filename: str = 'training_curves.png',
    ) -> None:
        """
        Render and save the training curve figure.

        Data persistence (CSV + JSON) is handled separately by
        ``save_history()``.  This method only produces the PNG; call
        ``save_history()`` first if you want both.

        Figure layout (3 panels): Loss | Accuracy | Learning Rate
        """
        out_dir   = os.path.join(self.exp_dir, self.PLOTS_DIR)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)

        epochs    = range(1, len(history['train_loss']) + 1)
        train_pct = [a * 100 for a in history['train_acc']]
        test_pct  = [a * 100 for a in history['test_acc']]

        fig = plt.figure(figsize=(15, 4))
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # ---- Panel 1: Loss ----
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(epochs, history['train_loss'], label='Train',
                 color='steelblue', linewidth=1.5)
        ax1.plot(epochs, history['test_loss'],  label='Test',
                 color='tomato',    linewidth=1.5)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        self._mark_best(ax1, epochs, history['test_loss'], mode='min', color='tomato')

        # ---- Panel 2: Accuracy ----
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, train_pct, label='Train', color='steelblue', linewidth=1.5)
        ax2.plot(epochs, test_pct,  label='Test',  color='tomato',    linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Acc (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        best_acc = self._mark_best(ax2, epochs, test_pct, mode='max', color='tomato')
        ax2.set_title(f'Accuracy  (best: {best_acc:.2f}%)' if best_acc is not None
                      else 'Accuracy')

        # ---- Panel 3: Learning Rate ----
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
        self.info(f'Training curves saved -> {save_path}')

    def plot_training_realtime(
            self,
            history: dict,
            epoch: int,
            total_epochs: int,
            interval: int = 10,
            title: str = 'Training Curves',
    ) -> None:
        """
        Save a plot every ``interval`` epochs and keep the CSV in sync.

        The CSV is overwritten on every call (always complete and resumable);
        plots are generated only at multiples of ``interval`` to reduce I/O.
        """
        # Always update CSV (available for real-time inspection).
        self.save_history(history)

        # Generate plot only at the specified interval or on the final epoch.
        if epoch % interval == 0 or epoch == total_epochs:
            is_final = (epoch == total_epochs)
            fname = ('training_curves.png' if is_final
                     else f'training_curves_e{epoch:04d}.png')
            self._plot_and_save(history, title=title, filename=fname)

    def _plot_and_save(
            self,
            history: dict,
            title: str,
            filename: str,
    ) -> None:
        """Plot and save without writing data (used by plot_training_realtime)."""
        out_dir   = os.path.join(self.exp_dir, self.PLOTS_DIR)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)

        epochs    = range(1, len(history['train_loss']) + 1)
        train_pct = [a * 100 for a in history['train_acc']]
        test_pct  = [a * 100 for a in history['test_acc']]

        fig = plt.figure(figsize=(15, 4))
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue', linewidth=1.5)
        ax1.plot(epochs, history['test_loss'],  label='Test',  color='tomato',    linewidth=1.5)
        ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)
        self._mark_best(ax1, epochs, history['test_loss'], mode='min', color='tomato')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, train_pct, label='Train', color='steelblue', linewidth=1.5)
        ax2.plot(epochs, test_pct,  label='Test',  color='tomato',    linewidth=1.5)
        best_acc = self._mark_best(ax2, epochs, test_pct, mode='max', color='tomato')
        ax2.set_title(f'Accuracy  (best: {best_acc:.2f}%)' if best_acc else 'Accuracy')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)')
        ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(epochs, history['lr'], color='mediumseagreen', linewidth=1.5)
        ax3.set_title('Learning Rate'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('LR')
        ax3.set_yscale('log'); ax3.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.info(f'Training curves saved -> {save_path}')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_best(ax, epochs, values, mode='min', color='tomato', markersize=6):
        """Mark the best point on a plot axis and annotate its value."""
        if not values:
            return None
        best_idx = int(np.argmin(values) if mode == 'min' else np.argmax(values))
        best_val = values[best_idx]
        best_ep  = list(epochs)[best_idx]
        ax.scatter(best_ep, best_val, color=color, zorder=5, marker='*', s=markersize ** 2)
        ax.annotate(
            f'{best_val:.3f}',
            xy=(best_ep, best_val),
            xytext=(6, 6), textcoords='offset points',
            fontsize=7.5, color=color,
        )
        return best_val
