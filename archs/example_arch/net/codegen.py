"""
example_arch static code generator (train phase).

PURPOSE
-------
``generate_code()`` is the *optional* interface that enables *static save
mode* in ``ResultSaver``.  When present, ``archs.has_codegen()`` returns
``True`` and the search result package contains a frozen ``net.py`` with no
runtime dependency on the ``archs`` package.

STATIC vs DYNAMIC SAVE MODE
-----------------------------
::

                        Static  (has generate_code)   Dynamic  (no generate_code)
    ──────────────────  ──────────────────────────    ──────────────────────────
    Runtime dep on      archs.example_arch            archs.example_arch
    archs package       modules only (stable)         net.Net (full decoder)
    ──────────────────  ──────────────────────────    ──────────────────────────
    Portability         High — ship net.py alone      Medium — needs archs pkg
    ──────────────────  ──────────────────────────    ──────────────────────────
    Decode logic        Frozen at search time          Always uses latest arch
    ──────────────────  ──────────────────────────    ──────────────────────────
    Maintenance cost    Must update codegen when Net   None
                        changes

Both modes expose the identical external signature to ``train.py``::

    Net(dropout=0.0, drop_connect_rate=0.0)

``ResultSaver._save_static`` writes the returned string to
``scripts/<name>/net.py`` and generates a matching ``__init__.py``.

DROPPATH HANDLING IN GENERATED CODE
-------------------------------------
``train.py`` updates ``net.drop_connect_rate`` every epoch.  The generated
code stores the global rate as ``self.drop_connect_rate`` and each block
call receives::

    drop_connect_rate=drop_connect_rate * <coeff>

where ``<coeff>`` is the block's pre-computed linear fraction
``i / total_blocks``.  This matches the dynamic ``Net`` exactly.

GENERATED FILE STRUCTURE
-------------------------
::

    # Auto-generated header comment
    import statements

    class Net(nn.Module):
        def __init__(self, dropout=0.0, drop_connect_rate=0.0):
            super().__init__()
            self.stem    = ConvBnAct(...)
            self.blocks  = nn.Sequential(
                MBConvBlock(...),   # stage_1_1
                MBConvBlock(...),   # stage_1_2
                ...
            )
            self.head    = nn.Sequential(...)
            self.drop_connect_rate = drop_connect_rate
            self.dropout    = nn.Dropout(...) if dropout > 0. else nn.Identity()
            self.classifier = nn.Linear(...)
            init_weight(self)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.head(x)
            x = self.dropout(x)
            return self.classifier(x)
"""

from __future__ import annotations

import textwrap
from datetime import datetime

import numpy as np

from archs.example_arch.modules.ops import make_divisible
from archs.example_arch.net.net import _DATASET_CFG


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_code(individual: np.ndarray, dataset: str) -> str:
    """
    Freeze a chromosome into a complete, self-contained ``net.py`` source.

    The returned string is PEP-8-compliant Python, importable as a module.
    It depends only on ``archs.example_arch.modules`` and
    ``template.func.init_weight`` — both stable, low-churn dependencies.

    Parameters
    ----------
    individual : np.ndarray, shape (num_blocks, 4)
        A single chromosome; each row is ``[e, k, se, f]``.
    dataset : str
        Must be a key in ``_DATASET_CFG``.

    Returns
    -------
    str
        Complete Python source ready to write to ``net.py``.

    Raises
    ------
    ValueError
        If *dataset* is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from archs.example_arch.genotypes import get_search_space
    >>> from archs.example_arch.net.population_initializer import initialize_population
    >>> ss  = get_search_space('cifar10')
    >>> pop = initialize_population(ss, pop_size=1)
    >>> src = generate_code(pop[0], 'cifar10')
    >>> 'class Net' in src and 'MBConvBlock' in src
    True
    """
    if dataset not in _DATASET_CFG:
        raise ValueError(
            f"generate_code: unsupported dataset {dataset!r}.  "
            f"Supported: {sorted(_DATASET_CFG)}"
        )

    cfg         = _DATASET_CFG[dataset]
    num_classes = cfg["num_classes"]
    stem_stride = cfg["stem_stride"]
    stem_ch     = cfg["stem_ch"]
    head_ch     = cfg["head_ch"]

    from archs.example_arch.genotypes import get_search_space
    search_space = get_search_space(dataset)
    names        = search_space["names"]
    num_blocks   = len(names)

    # Decode chromosome → one source line per MBConvBlock constructor call.
    block_lines, end_ch = _emit_block_lines(
        individual, names, search_space, num_blocks, stem_ch,
    )
    blocks_src = "\n".join(block_lines)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = textwrap.dedent(f"""\
        # =============================================================
        # Auto-generated by NAS framework
        # Arch    : example_arch
        # Dataset : {dataset}
        # Created : {timestamp}
        #
        # DO NOT EDIT MANUALLY.
        #   Retrain  : bash train.sh
        #   Inspect  : chromosome.json
        #   Regenerate: re-run search with the same chromosome
        # =============================================================

        from __future__ import annotations

        import torch
        import torch.nn as nn

        from archs.example_arch.modules import ConvBnAct, MBConvBlock
        from template.func import init_weight


        class Net(nn.Module):
            \"\"\"
            example_arch frozen network — auto-generated by NAS.

            All architecture parameters are hard-coded as literals.
            No runtime dependency on the dynamic chromosome-decoder.

            Parameters
            ----------
            dropout : float
                Dropout probability before the classifier.
            drop_connect_rate : float
                Stored for API compatibility with ``train.py``'s epoch hook.
                DropPath rates are fixed at construction time (linear schedule
                from 0 to ``drop_connect_rate`` across blocks).
            \"\"\"

            def __init__(
                self,
                dropout: float = 0.0,
                drop_connect_rate: float = 0.0,
            ) -> None:
                super().__init__()

                # ── Stem ─────────────────────────────────────────────────
                self.stem = ConvBnAct(3, {stem_ch}, kernel_size=3, stride={stem_stride})

                # ── Blocks ({num_blocks} blocks, dataset='{dataset}') ─────
                self.blocks = nn.Sequential(
        {blocks_src}
                )

                # ── Head ─────────────────────────────────────────────────
                self.head = nn.Sequential(
                    ConvBnAct({end_ch}, {head_ch}, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )

                # Preserved for train.py's per-epoch update hook.
                self.drop_connect_rate = drop_connect_rate

                self.dropout    = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
                self.classifier = nn.Linear({head_ch}, {num_classes})

                init_weight(self)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.stem(x)
                x = self.blocks(x)
                x = self.head(x)
                x = self.dropout(x)
                return self.classifier(x)
    """)

    return code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_block_lines(
    individual: np.ndarray,
    names: list[str],
    search_space: dict,
    num_blocks: int,
    stem_ch: int,
) -> tuple[list[str], int]:
    """
    Decode the chromosome and produce one source line per ``MBConvBlock``.

    Parameters
    ----------
    individual : np.ndarray
    names : list[str]
    search_space : dict
    num_blocks : int
    stem_ch : int

    Returns
    -------
    (lines, end_ch)
        ``lines``  : list of strings, each an ``MBConvBlock(...)`` call
                     indented for the ``nn.Sequential`` body (12-space indent).
        ``end_ch`` : output channel count of the last block (for the head).
    """
    lines: list[str] = []
    in_ch = stem_ch

    for i, name in enumerate(names):
        gene    = individual[i]
        e       = int(gene[0])
        k       = int(gene[1])
        se      = float(gene[2])
        f       = float(gene[3])

        base_ch = search_space[name][4][0]
        stride  = search_space[name][5]
        out_ch  = make_divisible(base_ch * f)

        # Linear drop-path coefficient: block 0 → 0.0, last block → 1.0.
        coeff = i / num_blocks

        lines.append(
            f"            MBConvBlock("
            f"in_ch={in_ch}, out_ch={out_ch}, stride={stride}, "
            f"expansion_ratio={e}, kernel_size={k}, "
            f"se_ratio={se!r}, "
            f"drop_connect_rate=drop_connect_rate * {coeff:.6f}"
            f"),  # {name}"
        )
        in_ch = out_ch

    return lines, in_ch


__all__ = ["generate_code"]
