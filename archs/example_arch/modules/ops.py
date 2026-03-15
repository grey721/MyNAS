"""
example_arch primitive operators and MBConvBlock.

OPERATOR HIERARCHY
------------------
::

    make_divisible      — EfficientNet-style channel rounding utility.
    ConvBnAct           — Conv2d + BatchNorm2d + SiLU (universal building brick).
    SqueezeExcitation   — Channel-attention gate (SE-Net style).
    MBConvBlock         — Mobile Inverted Bottleneck block (MBConv / EfficientNet).

BLOCK STRUCTURE (expansion_ratio > 1, se_ratio > 0, with residual)
--------------------------------------------------------------------
::

    ┌───────── Input x ──────────┐
    │                            │  ← identity shortcut
    │  Expand  PW  Conv          │    (only when stride=1 and in_ch == out_ch)
    │  (in_ch → mid_ch, 1×1)    │
    │                            │
    │  Depthwise  Conv           │
    │  (mid_ch, k×k, stride)    │
    │                            │
    │  Squeeze-and-Excitation    │
    │  (mid_ch → squeeze → mid_ch│
    │                            │
    │  Project  PW  Conv         │
    │  (mid_ch → out_ch, 1×1)   │
    │                            │
    │  DropPath (residual branch)│
    └──────────── + ─────────────┘
                  │
               Output

When ``expansion_ratio == 1`` the Expand step is skipped (``nn.Identity``).
When ``se_ratio == 0.0``      the SE    step is skipped (``nn.Identity``).
When ``stride > 1 or in_ch != out_ch`` the shortcut and DropPath are omitted.

DESIGN NOTES
------------
* All operators are pure ``nn.Module`` subclasses with no global mutable
  state — safe for ``DataParallel`` / ``DistributedDataParallel``.
* Activation: SiLU (Swish) throughout.  Hard-coded for simplicity; easy to
  parameterise by passing an ``act_layer`` factory if needed.
* BN hyper-parameters (``eps=1e-3, momentum=0.01``) follow the EfficientNet
  convention for numerical stability.
* ``drop_connect_rate`` is stored as an *instance attribute* of
  ``MBConvBlock``, not passed as a ``forward()`` argument.  This matches the
  framework's training convention: ``train.py`` updates
  ``net.drop_connect_rate`` each epoch and each block reads its own
  per-block coefficient at forward time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from template.drop import drop_path


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """
    Round *value* to the nearest multiple of *divisor*.

    This is the standard EfficientNet channel-rounding helper.  It guarantees
    the rounded result never falls below 90 % of the original, preventing
    excessive shrinkage on small channel counts.

    Parameters
    ----------
    value : float
        Raw (unrounded) channel count.
    divisor : int, optional
        Rounding granularity.  Default 8 suits most accelerators.
    min_value : int or None, optional
        Hard lower bound.  Defaults to *divisor*.

    Returns
    -------
    int

    Examples
    --------
    >>> make_divisible(33)
    32
    >>> make_divisible(37)
    40
    >>> make_divisible(3, divisor=8)
    8
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


# ---------------------------------------------------------------------------
# Primitive operators
# ---------------------------------------------------------------------------


class ConvBnAct(nn.Module):
    """
    Fused Conv2d → BatchNorm2d → Activation layer.

    Used for the network stem, the pointwise expand/project convolutions
    inside ``MBConvBlock``, and the head conv.  Pass ``act=False`` for
    project convolutions (which must not carry an activation before the
    residual addition).

    Parameters
    ----------
    in_ch : int
    out_ch : int
    kernel_size : int
    stride : int, optional
        Default 1.
    groups : int, optional
        Pass ``groups=in_ch`` for a depthwise convolution.  Default 1.
    act : bool, optional
        If ``True`` (default), append a SiLU activation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding,
            groups=groups, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel-attention gate.

    Computes a per-channel gating vector via global average pooling and a
    two-layer MLP bottleneck, then re-scales the input feature map.

    The squeeze dimension is ``max(1, round(in_ch * se_ratio))``; the
    ``max(1, ...)`` guard ensures at least one filter even for tiny se_ratio
    values.

    Parameters
    ----------
    in_ch : int
        Input (and output) channel count.
    se_ratio : float
        Squeeze ratio in ``(0, 1]``.  Typical: 0.25.

    References
    ----------
    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, in_ch: int, se_ratio: float) -> None:
        super().__init__()
        squeeze_ch = max(1, int(round(in_ch * se_ratio)))
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.reduce  = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, bias=True)
        self.act     = nn.SiLU(inplace=True)
        self.expand  = nn.Conv2d(squeeze_ch, in_ch, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.reduce(scale))
        scale = self.sigmoid(self.expand(scale))
        return x * scale


# ---------------------------------------------------------------------------
# MBConvBlock
# ---------------------------------------------------------------------------


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv).

    The canonical building block of EfficientNet and related families.
    Combines depthwise separable convolution with optional Squeeze-and-
    Excitation attention and a stochastic-depth residual connection.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count.
    stride : int
        Spatial stride (1 or 2).
    expansion_ratio : int
        Channel expansion multiplier: ``mid_ch = in_ch * expansion_ratio``.
        When 1 the expand conv is replaced by ``nn.Identity``.
    kernel_size : int
        Depthwise convolution kernel size (3, 5, or 7).
    se_ratio : float
        SE squeeze ratio.  0.0 disables SE.
    drop_connect_rate : float, optional
        Absolute DropPath probability for this block.  Computed by the caller
        as ``global_rate * (block_index / total_blocks)``.  0.0 disables
        DropPath.  Default 0.0.

    Notes
    -----
    The residual shortcut is applied only when ``stride == 1`` and
    ``in_ch == out_ch`` (identity conditions).  This matches the original
    MobileNetV2 / EfficientNet convention.

    ``drop_connect_rate`` is stored as a plain Python ``float`` attribute
    (not an ``nn.Parameter``).  ``train.py`` may update
    ``net.drop_connect_rate`` each epoch; individual blocks carry their own
    pre-computed coefficient and are not updated directly.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion_ratio: int,
        kernel_size: int,
        se_ratio: float,
        drop_connect_rate: float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch = make_divisible(in_ch * expansion_ratio)

        # ── Expand (pointwise) ────────────────────────────────────────────
        self.expand = (
            ConvBnAct(in_ch, mid_ch, kernel_size=1)
            if expansion_ratio != 1
            else nn.Identity()
        )

        # ── Depthwise ─────────────────────────────────────────────────────
        self.depthwise = ConvBnAct(
            mid_ch, mid_ch,
            kernel_size=kernel_size,
            stride=stride,
            groups=mid_ch,
        )

        # ── Squeeze-and-Excitation ────────────────────────────────────────
        self.se = (
            SqueezeExcitation(mid_ch, se_ratio=se_ratio)
            if se_ratio > 0.0
            else nn.Identity()
        )

        # ── Project (pointwise, no activation) ───────────────────────────
        self.project = ConvBnAct(mid_ch, out_ch, kernel_size=1, act=False)

        # ── Residual shortcut ─────────────────────────────────────────────
        # Applied only when spatial size and channel count are unchanged.
        self.has_skip          = stride == 1 and in_ch == out_ch
        self.drop_connect_rate = float(drop_connect_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.has_skip:
            if self.drop_connect_rate > 0.0 and self.training:
                out = drop_path(out, drop_prob=self.drop_connect_rate, training=True)
            out = out + residual

        return out


__all__ = [
    "make_divisible",
    "ConvBnAct",
    "SqueezeExcitation",
    "MBConvBlock",
]
