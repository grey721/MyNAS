import argparse
import importlib
import multiprocessing
import os
import sys
from multiprocessing import Process

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from load_dataset.loaders import AugLevel, get_train_loader, get_test_loader, DEFAULT_DATA_ROOTS
from template.tools import cal_flops_params
from utils.logger import Logger

parser = argparse.ArgumentParser('Training')

# Runtime environment
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_subprocess', action='store_true')

# Data
parser.add_argument('--dataset', type=str, required=True,
                    choices=list(DEFAULT_DATA_ROOTS),
                    help='Dataset name: cifar10 / cifar100 / imagenet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_level', type=str, default='basic',
                    choices=['none', 'basic', 'strong'],
                    help='Training augmentation strength: none / basic / strong')

# Model
parser.add_argument('--script_name', type=str, required=True)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--drop_connect_rate', type=float, default=0.2)

# Training
parser.add_argument('--total_epochs', type=int, default=600)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--warmup_lr', type=float, default=0.0001,
                    help='Initial learning rate during warmup')
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--grad_clip', type=float, default=5.0)

args = parser.parse_args()


class Trainer:
    def __init__(self, net: nn.Module, args):
        self.args = args
        self.file_id = args.script_name
        self.logger = Logger(name=f'Train_{self.file_id}')

        self.train_loader, self.test_loader = self._build_loaders(args)

        cudnn.benchmark = True

        self.net = net.cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.best_acc = 0.0

        # Each list is appended once per epoch; indices are therefore
        # 0-based epoch offsets.  All lists must stay in sync.
        self.history = {
            'train_loss': [],
            'test_loss':  [],
            'train_acc':  [],
            'test_acc':   [],
            'lr':         [],
        }

        flops, params = cal_flops_params(self.net, input_size=self._input_size(args.dataset))
        self.logger.info(f'FLOPs: {flops / 1e6:.2f}M, Params: {params / 1e6:.2f}M')
        self.logger.save_config(args)

    # ------------------------------------------------------------------
    # Output: data + curves + checkpoint
    # ------------------------------------------------------------------

    def _save_outputs(self, title: str) -> None:
        """
        Persist all training artefacts to disk.

        Called from the ``finally`` block of ``run()`` — must never raise.
        Each step runs in its own ``try/except`` so a failure in one cannot
        affect the others.

        Step order (intentional)
        ------------------------
        1. Raw data  — CSV + JSON via ``save_history()``.
           This is the primary output: the source of truth for all future
           visualisation and cross-model comparisons.  It runs first and
           is fully independent of matplotlib.

        2. Curve figure  — PNG via ``plot_training()``.
           Secondary, convenience output for a quick visual check during
           development.  Failure here never affects step 1.
        """
        if len(self.history['train_loss']) == 0:
            self.logger.warning('_save_outputs: no epoch data to save (0 epochs completed).')
            return

        # ---- Step 1: raw data (CSV + JSON) — primary, must succeed ----
        try:
            self.logger.save_history(self.history)
        except Exception as exc:
            self.logger.error(f'Failed to save training data: {exc}')

        # ---- Step 2: curve figure (PNG) — secondary, nice to have ----
        try:
            self.logger.plot_training(
                self.history,
                title=title,
                filename='training_curves.png',
            )
        except Exception as exc:
            self.logger.error(f'Failed to save training curves: {exc}')

        # ---- Final summary log line ----
        self.logger.info(f'--- Summary | Best Acc: {self.best_acc * 100:.2f}%')
        self.logger.info(f'            | Best Err: {(1 - self.best_acc) * 100:.2f}%')

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def run(self) -> float:
        """
        Run the full training schedule and return the best test accuracy.

        The epoch loop is wrapped in a try/except/finally block so that
        training data and curves are always written to disk, even when
        training is cut short by a KeyboardInterrupt or an unexpected
        exception.

        Interruption behaviour
        ----------------------
        KeyboardInterrupt
            Caught and logged as a warning; training stops cleanly and
            all data collected up to that point is saved.  The exception
            is *not* re-raised, so the process exits normally.
        Any other exception
            Logged as an error and re-raised so the caller can inspect
            the traceback.  The finally block still runs, so partial data
            is saved before propagation.
        """
        args         = self.args
        total_epoch  = args.total_epochs
        warmup_epoch = args.warmup_epochs
        lr           = args.lr
        warmup_lr    = args.warmup_lr

        optimizer = optim.SGD(
            self.net.parameters(),
            lr=lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epoch - warmup_epoch
        )

        w = len(str(total_epoch))
        prefix_w = 9 + w * 2
        try:
            for epoch in range(total_epoch):
                self.net.drop_connect_rate = args.drop_connect_rate * epoch / total_epoch

                # ---- Learning rate schedule ----
                if epoch < warmup_epoch:
                    cur_lr = warmup_lr + (lr - warmup_lr) / max(warmup_epoch - 1, 1) * epoch
                    for g in optimizer.param_groups:
                        g['lr'] = cur_lr
                else:
                    cur_lr = optimizer.param_groups[0]['lr']

                self.logger.info(f'Epoch [{epoch + 1:>{w}d}/{total_epoch}] | LR: {cur_lr:.6f}')
                self.history['lr'].append(cur_lr)

                # ---- Train ----
                t_loss, t_acc = self._train_epoch(optimizer)
                self.history['train_loss'].append(t_loss)
                self.history['train_acc'].append(t_acc)
                self.logger.info(
                    f'{"Train":>{prefix_w}} | Loss: {t_loss:.4f} | Acc: {t_acc * 100:6.2f} %'
                )

                if epoch >= warmup_epoch:
                    scheduler.step()

                # ---- Validate ----
                v_loss, v_acc = self._validate()
                self.history['test_loss'].append(v_loss)
                self.history['test_acc'].append(v_acc)
                self.logger.info(
                    f'{"Test":>{prefix_w}} | Loss: {v_loss:.4f} | Acc: {v_acc * 100:6.2f} %'
                )

        except KeyboardInterrupt:
            # Treat manual interruption as a clean (partial) completion.
            completed = len(self.history['train_loss'])
            self.logger.warning(
                f'Training interrupted by user after epoch {completed}/{total_epoch}.  '
                f'Saving collected data before exit.'
            )

        except Exception as exc:
            # Log and re-raise; the finally block still saves partial data.
            completed = len(self.history['train_loss'])
            self.logger.error(
                f'Training failed at epoch {epoch + 1}/{total_epoch} '
                f'({completed} epoch(s) completed): {exc}'
            )
            raise

        finally:
            # Runs unconditionally: normal finish, KeyboardInterrupt, or
            # any other exception.  Safe to call with 0 completed epochs.
            self._save_outputs(title=f'Training - {self.file_id}')

        return self.best_acc



    # ------------------------------------------------------------------
    # Per-epoch train / validate
    # ------------------------------------------------------------------

    def _train_epoch(self, optimizer):
        self.net.train()
        running_loss, total, correct = 0.0, 0, 0

        for inputs, labels in self.train_loader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.detach().max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()

        return running_loss / total, correct / total

    def _validate(self):
        self.net.eval()
        test_loss, total, correct = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, pred = outputs.max(1)
                total   += labels.size(0)
                correct += pred.eq(labels).sum().item()

        acc = correct / total
        if acc > self.best_acc:
            os.makedirs('./trained_models', exist_ok=True)
            torch.save(
                self.net.state_dict(),
                f'./trained_models/{self.file_id}_best.pt',
            )
            self.best_acc = acc

        return test_loss / total, acc

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_loaders(args):
        aug_level = AugLevel[args.aug_level.upper()]
        train = get_train_loader(
            args.dataset,
            batch_size=args.batch_size,
            aug_level=aug_level,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test = get_test_loader(
            args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        return train, test

    @staticmethod
    def _input_size(dataset: str):
        return (1, 3, 32, 32) if 'cifar' in dataset else (1, 3, 224, 224)


class Runner:
    """
    Thin wrapper that optionally runs training in a subprocess.

    use_subprocess=False (default): run in-process; easy to debug; ideal for
                                    single-model training.
    use_subprocess=True:            run in a child process; GPU memory is fully
                                    released on exit; suitable for NAS batch eval.
    """

    def __init__(self, use_subprocess: bool = False):
        self.use_subprocess = use_subprocess
        self._ensure_dirs()

    def run(self, gpu_id: int, net: nn.Module, args) -> None:
        if self.use_subprocess:
            p = Process(target=self._work, args=(gpu_id, net, args))
            p.start()
            p.join()
        else:
            self._work(gpu_id, net, args)

    @staticmethod
    def _work(gpu_id: int, net: nn.Module, args) -> None:
        torch.cuda.empty_cache()
        torch.cuda.set_device(gpu_id)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)

        trainer = Trainer(net, args)
        trainer.logger.info(
            f'GPU#{gpu_id}  PID:{os.getpid()}  '
            f'Worker:{multiprocessing.current_process().name}'
        )
        # Trainer.run() manages its own try/except/finally.
        # Non-KeyboardInterrupt exceptions are re-raised so subprocess-mode
        # callers can detect failure via p.exitcode != 0.
        trainer.run()

    @staticmethod
    def _ensure_dirs():
        for d in ['./logs', './scripts', './trained_models']:
            os.makedirs(d, exist_ok=True)


def _load_net(script_name: str, args) -> nn.Module:
    """Dynamically load the Net definition from scripts/."""
    module_name = f'scripts.{script_name}'
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return module.Net(dropout=args.dropout)


if __name__ == '__main__':
    net = _load_net(args.script_name, args)
    Runner(args.use_subprocess).run(args.gpu, net, args)
