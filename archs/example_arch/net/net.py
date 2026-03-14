"""
example_arch network: decodes a chromosome (individual) into an executable nn.Module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_DATASET_CFG: dict[str, dict] = {
    'cifar10': {'num_classes': 10, 'stem_stride': 1, 'stem_ch': 24},
    'cifar100': {'num_classes': 100, 'stem_stride': 1, 'stem_ch': 24},
    'imagenet': {'num_classes': 1000, 'stem_stride': 2, 'stem_ch': 32},
}


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round *v* up to the nearest multiple of *divisor* (minimum = *divisor*)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Net(nn.Module):
    """
    example_arch network.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


__all__ = ['Net']
