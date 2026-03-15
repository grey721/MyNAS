"""
example_arch — Canonical reference architecture for this NAS framework.

PURPOSE
-------
This package is the authoritative template for implementing a new arch.
Every design decision, naming convention, and interface contract is
documented here.  When adding a new arch:

    1. Copy this directory and rename it (e.g. ``my_arch/``).
    2. Adapt ``genotypes/`` to describe your search space.
    3. Adapt ``modules/ops.py`` to implement your block primitive.
    4. Adapt ``net/net.py`` to decode chromosomes into ``nn.Module``.
    5. Adapt ``net/population_initializer.py`` if your gene structure changes.
    6. Optionally implement ``net/codegen.py`` to enable static save mode.

PACKAGE LAYOUT
--------------
::

    example_arch/
    ├── __init__.py                   ← Framework entry point (this file).
    │
    ├── genotypes/
    │   ├── __init__.py               ← get_search_space() dispatcher.
    │   ├── cifar.py                  ← CIFAR-10 / CIFAR-100 search space.
    │   └── imagenet.py               ← ImageNet search space.
    │
    ├── modules/
    │   ├── __init__.py               ← Public re-exports.
    │   └── ops.py                    ← Primitive operators + MBConvBlock.
    │
    └── net/
        ├── __init__.py               ← Public re-exports.
        ├── net.py                    ← Net: runtime decoder (search phase).
        ├── population_initializer.py ← initialize_population() + sample_gene().
        └── codegen.py                ← generate_code(): static freezer (train phase).

═══════════════════════════════════════════════════════════════════════════════
REQUIRED INTERFACE  —  archs.load_arch() validates all three are present.
═══════════════════════════════════════════════════════════════════════════════

``Net``
    An ``nn.Module`` subclass.  Decodes a chromosome at runtime.
    Constructor signature (must match exactly)::

        Net(individual, dataset, dropout=0.0, drop_connect_rate=0.0)

    ``individual``        : array-like (num_blocks, NUM_GENE_PER_BLOCK), dtype=object
    ``dataset``           : str
    ``dropout``           : float, default 0.0
    ``drop_connect_rate`` : float, default 0.0

``initialize_population``
    Callable with signature::

        initialize_population(search_space: dict, pop_size: int)
            -> np.ndarray  shape (pop_size, num_blocks, NUM_GENE_PER_BLOCK)

    Called by ``Searcher._initialize_population()`` at the start of search.

``get_search_space``
    Callable with signature::

        get_search_space(dataset: str) -> dict

    The dict must contain ``'names'``: an ordered list of block names.
    All other keys map a block name to its candidate-value lists.

═══════════════════════════════════════════════════════════════════════════════
OPTIONAL INTERFACE  —  unlocks additional framework features when present.
═══════════════════════════════════════════════════════════════════════════════

``generate_code``
    Callable with signature::

        generate_code(individual: np.ndarray, dataset: str) -> str

    Returns the complete Python source of a frozen ``net.py``.
    ``archs.has_codegen()`` returns ``True`` when this is present, causing
    ``ResultSaver`` to use *static mode* (self-contained saved package with
    no runtime dependency on the ``archs`` package).
    Without it, ``ResultSaver`` falls back to *dynamic mode* (a wrapper
    ``Net`` that subclasses ``arch.Net`` and injects the chromosome at init).

``sample_gene``
    Callable with signature::

        sample_gene(search_space: dict, block_name: str) -> list

    Returns one randomly sampled gene vector for the named block.
    Used by ``Searcher.mute()`` for per-block mutation.  Without it,
    ``mute()`` falls back to uniform sampling over each candidate list,
    which may violate arch-specific structural constraints.

═══════════════════════════════════════════════════════════════════════════════
GENE ENCODING  —  [e, k, se, f]
═══════════════════════════════════════════════════════════════════════════════

Each block's gene is a vector of length ``NUM_GENE_PER_BLOCK = 4``:

    Index  Name  Type   Meaning                    Typical candidates
    ─────  ────  ─────  ─────────────────────────  ──────────────────
      0    e     int    Inverted-bottleneck ratio   [1, 3, 6]
      1    k     int    Depthwise kernel size       [3, 5, 7]
      2    se    float  SE squeeze ratio            [0.0, 0.25]
      3    f     float  Channel width multiplier    [0.75, 1.0, 1.25]

═══════════════════════════════════════════════════════════════════════════════
SEARCH-SPACE ENTRY LAYOUT
═══════════════════════════════════════════════════════════════════════════════

Each per-block entry in the search-space dict is a six-element list::

    search_space[block_name] = [
        e_cands,    # list[int]   — expansion-ratio candidates       (gene idx 0)
        k_cands,    # list[int]   — depthwise kernel-size candidates  (gene idx 1)
        se_cands,   # list[float] — SE squeeze-ratio candidates       (gene idx 2)
        f_cands,    # list[float] — channel scale-factor candidates   (gene idx 3)
        [base_ch],  # list[int]   — singleton: base output channels   (not sampled)
        stride,     # int         — spatial stride, 1 or 2            (not sampled)
    ]

Indices 0–3 are sampled by ``initialize_population`` and ``sample_gene``.
Index 4 is a singleton list; access as ``search_space[name][4][0]``.
Index 5 is a scalar read directly by ``Net`` and ``generate_code``.

═══════════════════════════════════════════════════════════════════════════════
STATIC vs DYNAMIC SAVE MODE
═══════════════════════════════════════════════════════════════════════════════

                           Static (has generate_code)   Dynamic (no generate_code)
    ─────────────────────  ───────────────────────────  ──────────────────────────
    Runtime dep on archs   No                           Yes (arch.Net imported)
    Portability            High — ship net.py alone     Medium — needs archs pkg
    Decode logic freshness Frozen at search time        Always uses latest arch
    Codegen maintenance    Required                     None

Both modes expose the identical external ``Net(dropout, drop_connect_rate)``
signature to ``train.py``.
"""

from archs.example_arch.genotypes import get_search_space
from archs.example_arch.net import Net, initialize_population, generate_code
from archs.example_arch.net.population_initializer import sample_gene

__all__ = [
    # Required interface
    "Net",
    "initialize_population",
    "get_search_space",
    # Optional interface
    "generate_code",
    "sample_gene",
]
