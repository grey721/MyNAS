"""
SynFlow zero-cost proxy.

Reference
---------
Tanaka et al., "Pruning Neural Networks without Any Data by Iteratively
Conserving Synaptic Flow", NeurIPS 2020.

Method
------
Computes the product of parameter gradients under an all-ones input without
any real data, measuring the information flow capacity through the network's
synaptic pathways.
    score = sum(|theta o dR/d_theta|),  R = prod(|theta|)

Advantages
----------
- Completely data-free (no dependency on the input distribution).
- Very fast to evaluate.

Score direction: higher is better.

Note
----
SynFlow requires all parameters to be non-negative to produce meaningful
scores.  This is achieved by temporarily taking the absolute value of all
parameters.  A ``deepcopy`` of the network is used to avoid polluting the
original weights.

Usage
-----
::

    from ea.proxy.synflow import SynFlowProxy

    proxy = SynFlowProxy()
    score = proxy.score(net, batch)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ea.proxy.base import BaseProxy


class SynFlow(BaseProxy):
    """
    SynFlow proxy: data-free synaptic flow score.

    The ``x`` tensor in *batch* is used only to determine the input shape;
    its values are ignored.
    """

    def _compute(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        x, _ = batch
        # Replace the real input with an all-ones tensor of the same shape.
        ones = torch.ones_like(x)

        # deepcopy to avoid modifying the original network parameters.
        net_copy = copy.deepcopy(net)

        # Temporarily set all parameters to their absolute values (SynFlow requirement).
        for p in net_copy.parameters():
            p.data.abs_()

        net_copy.zero_grad()
        net_copy.train()

        output = net_copy(ones)
        # R = prod(|theta|); approximated by summing the output directly.
        loss = output.sum()
        loss.backward()

        score = 0.0
        for p in net_copy.parameters():
            if p.grad is not None:
                score += (p.data * p.grad.abs()).sum().item()

        return score

    def __repr__(self) -> str:
        return "SynFlow()"


__all__ = ['SynFlow']
