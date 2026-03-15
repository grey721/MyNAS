"""
example_arch modules sub-package.

Re-exports the public operator API so callers can write::

    from archs.example_arch.modules import MBConvBlock, ConvBnAct

without coupling to the internal file layout.
"""

from archs.example_arch.modules.ops import (
    ConvBnAct,
    MBConvBlock,
    SqueezeExcitation,
    make_divisible,
)

__all__ = [
    "make_divisible",
    "ConvBnAct",
    "SqueezeExcitation",
    "MBConvBlock",
]
