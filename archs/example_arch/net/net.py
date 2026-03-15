"""
example_arch dynamic network decoder (search phase).

PURPOSE
-------
``Net`` decodes a chromosome into a runnable ``nn.Module`` at *runtime*
during the search phase.  It is instantiated once per individual per
generation inside ``Evaluator.evaluate()``.  No code generation occurs here.

CHROMOSOME DECODING
-------------------
``individual[i]`` is a gene vector ``[e, k, se, f]`` (length 4).

    e  → MBConvBlock(expansion_ratio=e)
    k  → MBConvBlock(kernel_size=k)
    se → MBConvBlock(se_ratio=se)
    f  → out_ch = make_divisible(base_ch × f)

The input channel count for block *i* is the output channel count of
block *i−1* (or ``stem_ch`` for the first block).  ``in_ch`` is therefore
fully determined by the chromosome and needs no explicit encoding.

NETWORK STRUCTURE
-----------------
::

    Stem      ConvBnAct(3 → stem_ch, k=3, stride=stem_stride)
    Blocks    nn.Sequential(*[MBConvBlock(...) for each gene])
    Head      ConvBnAct(end_ch → 1280, k=1)
              → AdaptiveAvgPool2d(1) → Flatten
    Dropout   nn.Dropout(p=dropout)
    Classifier nn.Linear(1280 → num_classes)

DROPPATH SCHEDULE
-----------------
Block *i* out of *N* total blocks receives::

    drop_rate_i = drop_connect_rate × (i / N)

Rate 0 for the first block, rising linearly to ``drop_connect_rate`` for
the last.  Rates are fixed at construction time.  ``train.py`` may write
``net.drop_connect_rate`` each epoch; the attribute is preserved for API
compatibility but does not retroactively update block rates.

DATASET CONFIGURATION
---------------------
``_DATASET_CFG`` maps dataset name → stem/head config.  Add entries here
when supporting additional datasets; no other file needs to change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from archs.example_arch.modules import ConvBnAct, MBConvBlock, make_divisible
from template.func import init_weight

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# Each entry: stem_stride, stem_ch, head_ch, num_classes.
_DATASET_CFG: dict[str, dict] = {
    "cifar10":  {"stem_stride": 1, "stem_ch": 32, "head_ch": 1280, "num_classes": 10},
    "cifar100": {"stem_stride": 1, "stem_ch": 32, "head_ch": 1280, "num_classes": 100},
    "imagenet": {"stem_stride": 2, "stem_ch": 32, "head_ch": 1280, "num_classes": 1000},
}

# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------


class Net(nn.Module):
    """
    example_arch runtime decoder network.

    Instantiated once per candidate during the search phase.  All architecture
    parameters are derived from *individual* and *dataset* at ``__init__``
    time; ``forward`` contains no conditional logic.

    Parameters
    ----------
    individual : array-like, shape (num_blocks, 4), dtype=object
        Chromosome produced by ``initialize_population``.
        Row *i* is ``[e, k, se, f]`` for block *i*.
    dataset : str
        Dataset name.  Must be a key in ``_DATASET_CFG``.
    dropout : float, optional
        Dropout probability before the classifier.  Default 0.0.
    drop_connect_rate : float, optional
        Global DropPath ceiling linearly distributed across blocks.
        Stored as ``self.drop_connect_rate`` for ``train.py`` API compat.
        Default 0.0.

    Raises
    ------
    ValueError
        If *dataset* is not supported.
    """

    def __init__(
        self,
        individual,
        dataset: str,
        dropout: float = 0.0,
        drop_connect_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if dataset not in _DATASET_CFG:
            raise ValueError(
                f"Net: unsupported dataset {dataset!r}.  "
                f"Supported: {sorted(_DATASET_CFG)}"
            )

        cfg         = _DATASET_CFG[dataset]
        num_classes = cfg["num_classes"]
        stem_ch     = cfg["stem_ch"]
        head_ch     = cfg["head_ch"]
        stem_stride = cfg["stem_stride"]

        from archs.example_arch.genotypes import get_search_space
        search_space = get_search_space(dataset)
        names        = search_space["names"]
        num_blocks   = len(names)

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = ConvBnAct(3, stem_ch, kernel_size=3, stride=stem_stride)

        # ── Blocks ────────────────────────────────────────────────────────
        blocks: list[nn.Module] = []
        in_ch = stem_ch

        for i, name in enumerate(names):
            gene = individual[i]
            e    = int(gene[0])
            k    = int(gene[1])
            se   = float(gene[2])
            f    = float(gene[3])

            base_ch = search_space[name][4][0]  # singleton list → scalar
            stride  = search_space[name][5]      # fixed per block; not sampled
            out_ch  = make_divisible(base_ch * f)

            # Linearly scale DropPath: block 0 gets 0, last block gets the ceiling.
            block_dp = drop_connect_rate * i / num_blocks

            blocks.append(
                MBConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    stride=stride,
                    expansion_ratio=e,
                    kernel_size=k,
                    se_ratio=se,
                    drop_connect_rate=block_dp,
                )
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            ConvBnAct(in_ch, head_ch, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # ── Classifier ────────────────────────────────────────────────────
        # Preserve drop_connect_rate as an attribute: train.py may overwrite
        # it each epoch (API compatibility; does not affect block rates).
        self.drop_connect_rate = drop_connect_rate
        self.dropout    = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(head_ch, num_classes)

        init_weight(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["Net", "_DATASET_CFG"]
