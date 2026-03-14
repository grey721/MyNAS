"""
example_arch code generator.

Responsibility
--------------
Freeze a single chromosome (individual) into a complete Python source string
that can be written to ``scripts/<n>/net.py`` and loaded directly by ``train.py``
via ``importlib``, with no runtime dependency on the dynamic decode path.

Dynamic Net vs static net.py
------------------------------
    net.py (``Net`` class)              codegen.py (``generate_code``)
    ----------------------------        -----------------------------------
    Used during the search phase.       Used during the training phase.
    ``Net(individual, dataset)``        ``Net(dropout, drop_connect_rate)``
    Decodes the chromosome at runtime.  Architecture is hard-coded; no individual arg.
    Depends on the archs package.       Only depends on archs.example_arch.modules (stable).

Generated code structure
------------------------
    # Auto-generated ... (header comment)
    import statements
    class Net(nn.Module):
        def __init__(self, dropout=0.2, drop_connect_rate=0.0):
            self.stem    = ...   (hard-coded parameters)
            self.blocks  = nn.Sequential(
                CSBConvBlock(in_ch=..., out_ch=..., ...),  # block name comment
                nn.Identity(),                              # skip block
                ...
            )
            self.head       = ...
            self.classifier = ...
            init_weight(self)
        def forward(self, x): ...

Handling drop_connect_rate
---------------------------
    ``train.py`` updates ``net.drop_connect_rate`` each epoch:
        ``self.net.drop_connect_rate = args.drop_connect_rate * epoch / total_epoch``
    Each block's actual DropPath rate must therefore be read dynamically in
    ``forward``, not fixed at ``__init__`` time.
    The generated code achieves this by storing each block's relative coefficient
    ``dp_ratio`` at ``__init__`` and multiplying by ``self.drop_connect_rate`` in
    ``forward``. This requires ``CSBConvBlock`` to accept ``drop_connect_rate`` in
    its ``forward`` signature.

    Simplified approach (current): DropPath rates are distributed linearly in
    ``__init__`` using the initial ``drop_connect_rate``. The
    ``net.drop_connect_rate`` attribute is preserved for API compatibility with
    ``train.py`` but does not affect individual block rates after construction.
    This matches the behaviour of the original NAS codebase.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

import numpy as np

from archs.example_arch.net.net import _DATASET_CFG, _make_divisible


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_code(individual: np.ndarray, dataset: str) -> str:
    """
    Freeze a chromosome into a complete ``net.py`` source string.

    Parameters
    ----------
    individual:
        Shape (num_blocks, 6); a single chromosome.
    dataset:
        Dataset name; determines stem / head configuration and number of classes.

    Returns
    -------
    str
        Full Python source code ready to be written to ``net.py``.

    Raises
    ------
    ValueError
        If *dataset* is not supported.
    """
    if dataset not in _DATASET_CFG:
        raise ValueError(
            f"generate_code: unsupported dataset {dataset!r}. "
            f"Available: {list(_DATASET_CFG)}"
        )

    cfg = _DATASET_CFG[dataset]
    num_classes = cfg['num_classes']
    stem_stride = cfg['stem_stride']
    stem_ch = cfg['stem_ch']

    from archs.example_arch.genotypes import get_search_space
    search_space = get_search_space(dataset)
    names = search_space['names']
    total_blocks = len(names)

    # ------------------------------------------------------------------
    # 1. Decode chromosome block by block; emit block instantiation lines.
    # ------------------------------------------------------------------
    block_lines: list[str] = []
    in_ch = stem_ch

    for idx, name in enumerate(names):
        gene = individual[idx]
        s = float(gene[0])
        e = int(gene[1])
        b = int(gene[2])
        k = int(gene[3])
        se_ratio = float(gene[4])
        f = float(gene[5])

        base_ch = search_space[name][6][0]
        stride = search_space[name][7]
        out_ch = _make_divisible(base_ch * f)

        # dp_ratio: this block's linear coefficient within [0, drop_connect_rate].
        dp_ratio = idx / total_blocks

        if s == 0 and stride == 1 and in_ch == out_ch:
            block_lines.append(f'            nn.Identity(),                   # {name} (skip)')
        else:
            effective_s = max(s, 1.0) if s == 0 else s
            block_lines.append(
                f'            Block('
                f'in_ch={in_ch}, out_ch={out_ch}, stride={stride}, '
                f's={effective_s}, e={e}, b={b}, k={k}, '
                f'se_ratio={se_ratio}, '
                f'drop_connect_rate=drop_connect_rate * {dp_ratio:.6f}'
                f'),  # {name}'
            )

        in_ch = out_ch

    blocks_body = '\n'.join(block_lines)

    # ------------------------------------------------------------------
    # 2. Assemble the full source string.
    # ------------------------------------------------------------------
    code = textwrap.dedent(f'''\
        # =============================================================
        # Auto-generated by NAS -- arch: example_arch, dataset: {dataset}
        # Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # Do NOT edit manually.
        # To retrain: bash train.sh
        # To inspect chromosome: chromosome.json
        # =============================================================

        import torch
        import torch.nn as nn

        from archs.example_arch.modules import CSBConvBlock
        from template.func import init_weight


        class Net(nn.Module):
            """
            example_arch frozen network (auto-generated by NAS).

            Parameters
            ----------
            dropout:
                Classifier head dropout probability.
            drop_connect_rate:
                DropPath probability ceiling; train.py updates this attribute
                dynamically each epoch.
            """

            def __init__(
                self,
                dropout: float = 0.2,
                drop_connect_rate: float = 0.0,
            ) -> None:
                super().__init__()

                # --- Stem ---
                self.stem = nn.Sequential(
                    nn.Conv2d(3, {stem_ch}, kernel_size=3, stride={stem_stride}, padding=1, bias=False),
                    nn.BatchNorm2d({stem_ch}, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                )

                # --- Blocks ({total_blocks} blocks, dataset={dataset}) ---
                self.blocks = nn.Sequential(
        {blocks_body}
                )

                # --- Head ---
                self.head = nn.Sequential(
                    nn.Conv2d({in_ch}, 1280, kernel_size=1, bias=False),
                    nn.BatchNorm2d(1280, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )

                self.drop_connect_rate = drop_connect_rate  # updated by train.py each epoch
                self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                self.classifier = nn.Linear(1280, {num_classes})

                init_weight(self)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.stem(x)
                x = self.blocks(x)
                x = self.head(x)
                x = self.dropout(x)
                return self.classifier(x)
    ''')

    return code


__all__ = ['generate_code']
