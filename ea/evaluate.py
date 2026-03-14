from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from archs.arch import Net
from load_dataset.loaders import AugLevel, get_nas_loader
from template.tools import cal_flops_params


_DATASET_META: dict[str, dict] = {
    'cifar10':  {'input_shape': (1, 3, 32, 32),   'num_workers': 4},
    'cifar100': {'input_shape': (1, 3, 32, 32),   'num_workers': 4},
    'imagenet': {'input_shape': (1, 3, 224, 224), 'num_workers': 32},
}


def _logdet(K: np.ndarray) -> float:
    _, ld = np.linalg.slogdet(K)
    return float(ld)


@contextmanager
def _relu_hooks(net: nn.Module, K_buf: list[np.ndarray]) -> Iterator[None]:
    """
    为 net 中所有 ReLU 注册 forward hook，将 activation kernel 累加到 K_buf[0]。
    使用 contextmanager 确保 hook 在退出时一定被移除。
    """
    handles = []

    def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        x = inp[0] if isinstance(inp, tuple) else inp
        x = x.view(x.size(0), -1)
        binary = (x > 0).float()
        K_buf[0] = (
            K_buf[0]
            + (binary @ binary.t()).cpu().numpy()
            + ((1.0 - binary) @ (1.0 - binary.t())).cpu().numpy()
        )

    for module in net.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(_hook))

    try:
        yield   # 交给上下文管理器，退出后执行 finally
    finally:
        for h in handles:
            h.remove()


class Evaluator:
    OBJ_NAMES = ('fitness', 'err', 'n_parameters', 'n_flops')

    def __init__(
        self,
        dataset: str,
        batch_size: int,
        # random_seed: int = 42,
    ) -> None:
        if dataset not in _DATASET_META:
            raise ValueError(
                f'不支持的数据集: {dataset!r}，可选: {list(_DATASET_META)}'
            )

        cfg = _DATASET_META[dataset]
        self.dataset = dataset
        self.batch_size_search = batch_size
        self.input_shape: tuple[int, ...] = cfg['input_shape']

        # generator = torch.Generator().manual_seed(random_seed)
        loader = get_nas_loader(
            dataset=dataset,
            batch_size=batch_size,
            aug_level=AugLevel.NONE,    # ZC-proxy 不需要数据增强
            num_workers=cfg['num_workers'],
            pin_memory=True,
            # generator=generator,
        )
        self._data_iter = iter(loader)

    def evaluate(self, population: list) -> np.ndarray:
        pop_size = len(population)
        fitness_matrix = np.zeros((pop_size, len(self.OBJ_NAMES)), dtype=np.float32)

        for i, indi in enumerate(population):
            net = Net(indi, self.dataset)
            err, n_params, n_flops = self._evaluate(net)
            fitness_matrix[i, 0] = self._cal_fitness(err, n_params, n_flops)
            fitness_matrix[i, 1] = err
            fitness_matrix[i, 2] = n_params
            fitness_matrix[i, 3] = n_flops

        return fitness_matrix
    
    @staticmethod
    def _cal_fitness(err: float, n_parameters: int, n_flops: int) -> float:
        return -err
    
    def _evaluate(self, net: Net) -> tuple[float, int, int]:
        net = net.cuda()
        batch_size = self.batch_size_search
        K_buf = [np.zeros((batch_size, batch_size), dtype=np.float64)]

        x, target = next(self._data_iter)
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        try:
            with _relu_hooks(net, K_buf):
                net(x)

            naswot_score = _logdet(K_buf[0])
            err = -naswot_score  # err 越小 → 线性区域多样性越高

            n_flops, n_params = cal_flops_params(net, input_size=self.input_shape)

        finally:
            del x, target

        return err, n_params, n_flops