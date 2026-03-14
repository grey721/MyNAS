"""
Evaluator: decodes each chromosome in a population into a network and
computes its fitness metrics.

Responsibilities
----------------
  ``Evaluator``      : manages data loading, FLOPs / Params counting,
                       and fitness aggregation.
  ``BaseProxy``      : computes a single zero-cost proxy score.
  ``CompositeProxy`` : combines multiple proxy scores via weighted summation.

Usage examples
--------------
    # Single proxy
    from ea.proxy.naswot import NasWotProxy

    evaluator = Evaluator('example_arch', 'cifar10', batch_size=128,
                          proxy=NasWotProxy(batch_size=128))

    # Different proxy
    from ea.proxy.synflow import SynFlowProxy

    evaluator = Evaluator('example_arch', 'cifar10', batch_size=128,
                          proxy=SynFlowProxy())

    # Composite proxy (weighted)
    from ea.proxy import CompositeProxy
    from ea.proxy.naswot  import NasWotProxy
    from ea.proxy.synflow import SynFlowProxy

    proxy = CompositeProxy(
        proxies=[NasWotProxy(batch_size=128), SynFlowProxy()],
        weights=[0.7, 0.3],
    )
    evaluator = Evaluator('example_arch', 'cifar10', batch_size=128, proxy=proxy)
"""

from __future__ import annotations

import itertools

import numpy as np
import torch.nn as nn

from archs import load_arch
from ea.proxy.base import BaseProxy
from ea.proxy.naswot import NASWOT
from load_dataset.loaders import AugLevel, get_nas_loader
from template.tools import cal_flops_params

_DATASET_META: dict[str, dict] = {
    'cifar10': {'input_shape': (1, 3, 32, 32), 'num_workers': 4},
    'cifar100': {'input_shape': (1, 3, 32, 32), 'num_workers': 4},
    'imagenet': {'input_shape': (1, 3, 224, 224), 'num_workers': 32},
}

OBJ_NAMES = ('fitness', 'err', 'n_parameters', 'n_flops')

COL_FITNESS = 0
COL_ERR = 1
COL_PARAMS = 2
COL_FLOPS = 3


class Evaluator:
    """
    Population evaluator.

    Parameters
    ----------
    arch_name:
        Subdirectory name under ``archs/``, e.g. ``'example_arch'``.
    dataset:
        Dataset name, e.g. ``'cifar10'``.
    batch_size:
        Batch size used during NAS evaluation.
    proxy:
        A ``BaseProxy`` instance to use for scoring.
        Instantiate and pass the proxy explicitly::

            from ea.proxy.naswot  import NasWotProxy
            from ea.proxy.synflow import SynFlowProxy

            Evaluator(..., proxy=NasWotProxy(batch_size=128))
            Evaluator(..., proxy=SynFlowProxy())

        Defaults to ``NasWotProxy(batch_size=batch_size)``.
    """

    def __init__(
            self,
            arch_name: str,
            dataset: str,
            batch_size: int,
            proxy: BaseProxy | None = None,
    ) -> None:
        if dataset not in _DATASET_META:
            raise ValueError(
                f"Unsupported dataset: {dataset!r}. "
                f"Available: {list(_DATASET_META)}"
            )

        # Load arch and extract the Net class.
        arch_module = load_arch(arch_name)
        self.Net = arch_module.Net
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_shape = _DATASET_META[dataset]['input_shape']

        # Default to NasWotProxy when no proxy is supplied.
        self.proxy: BaseProxy = proxy if proxy is not None else NASWOT(batch_size=batch_size)

        # Data loader wrapped in itertools.cycle so it never exhausts across generations.
        cfg = _DATASET_META[dataset]
        loader = get_nas_loader(
            dataset=dataset,
            batch_size=batch_size,
            aug_level=AugLevel.NONE,
            num_workers=cfg['num_workers'],
            pin_memory=True,
        )
        self._data_iter = itertools.cycle(loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, population) -> np.ndarray:
        """
        Evaluate an entire population.

        Parameters
        ----------
        population:
            List or ndarray of chromosomes, each with shape
            ``(num_blocks, num_params)``.

        Returns
        -------
        np.ndarray
            Shape ``(pop_size, 4)``, ``dtype=float32``.
            Column order: ``[fitness, err, n_params, n_flops]``
            (constants ``COL_FITNESS``, ``COL_ERR``, ``COL_PARAMS``, ``COL_FLOPS``).
        """
        pop_size = len(population)
        fitness_matrix = np.zeros((pop_size, len(OBJ_NAMES)), dtype=np.float32)
        for i, indi in enumerate(population):

            net = self.Net(indi, self.dataset)

            proxy_score = self._score(net)
            n_flops, n_params = cal_flops_params(net, input_size=self.input_shape)
           
            fitness_matrix[i, COL_FITNESS] = proxy_score
            fitness_matrix[i, COL_ERR] = -proxy_score  # err: lower is better
            fitness_matrix[i, COL_PARAMS] = n_params
            fitness_matrix[i, COL_FLOPS] = n_flops

        return fitness_matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self, net: nn.Module) -> float:
        """Fetch the next batch and compute the proxy score."""
        batch = next(self._data_iter)  # (x, target) on CPU
        return self.proxy.score(net, batch)

    def __repr__(self) -> str:
        return (
            f"Evaluator(arch={self.Net.__module__}, "
            f"dataset={self.dataset!r}, "
            f"proxy={self.proxy!r})"
        )
