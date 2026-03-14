"""
example_arch Block implementation.

"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# example_arch Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    example_arch Block.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise x


__all__ = ['Block']
