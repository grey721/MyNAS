"""
NAS-WOT (Neural Architecture Search Without Training) zero-cost proxy.

Reference
---------
Mellor et al., "Neural Architecture Search without Training", ICML 2021.

Method
------
Measures the diversity of a network's linear regions via the log-determinant
of the ReLU activation kernel matrix.  Higher log|K| indicates stronger
discriminative capacity over the input batch.

Score direction: higher is better.

Usage
-----
::

    from ea.proxy.naswot import NasWotProxy

    proxy = NasWotProxy(batch_size=128)
    score = proxy.score(net, batch)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from ea.proxy.base import BaseProxy


def _logdet(K: torch.Tensor) -> float:
    # slogdet on GPU (torch >= 1.9); fall back to numpy on CPU
    try:
        _, ld = torch.linalg.slogdet(K)
        return float(ld)
    except Exception:
        _, ld = np.linalg.slogdet(K.cpu().numpy())
        return float(ld)


@contextmanager
def _relu_hooks(
    net: nn.Module,
    K_buf: list[torch.Tensor],
) -> Iterator[None]:
    """
    Register forward hooks on all ReLU modules in *net*.

    K is accumulated entirely on GPU; no per-hook CPU transfer.
    A context manager guarantees hook removal on exit.
    """
    handles = []

    def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        x = inp[0] if isinstance(inp, tuple) else inp
        x = x.view(x.size(0), -1)
        binary = (x > 0).float()
        # stays on GPU — no .cpu() / .numpy() here
        K_buf[0] = (
            K_buf[0]
            + binary @ binary.t()
            + (1.0 - binary) @ (1.0 - binary).t()
        )

    for module in net.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


class NASWOT(BaseProxy):
    """
    NAS-WOT proxy: log-determinant of the ReLU activation kernel matrix.

    Parameters
    ----------
    batch_size:
        Evaluation batch size; used to pre-allocate the kernel matrix.
        Must match the actual batch size passed by the ``Evaluator``.
    """

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def _compute(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        x, _ = batch
        # initialise K on the same device as x (already on GPU via base.py)
        K_buf = [torch.zeros(x.size(0), x.size(0),
                             dtype=torch.float64, device=x.device)]
        
        with torch.no_grad():
            with _relu_hooks(net, K_buf):
                net(x)

        return _logdet(K_buf[0])

    def __repr__(self) -> str:
        return f"NasWotProxy(batch_size={self.batch_size})"


__all__ = ['NASWOT']
