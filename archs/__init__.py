"""
archs package entry point.

Exposes two factory/utility functions: ``load_arch`` and ``has_codegen``.

Required interface (every arch must implement)
-----------------------------------------------
    Net                   : type[nn.Module]
        ``Net(individual, dataset, dropout=0.0, drop_connect_rate=0.0) -> nn.Module``
        Used during the search phase; decodes a chromosome into a network at runtime.

    initialize_population : Callable[[dict, int], np.ndarray]
        ``initialize_population(search_space, pop_size) -> ndarray (pop_size, num_blocks, num_params)``

    get_search_space      : Callable[[str], dict]
        ``get_search_space(dataset) -> dict``
        Key ``'names'`` holds an ordered list of block names; remaining keys map
        block names to their candidate-value lists.

Optional interface (enables the static save mode in ResultSaver)
-----------------------------------------------------------------
    generate_code         : Callable[[np.ndarray, str], str]
        ``generate_code(individual, dataset) -> str``  (full net.py source)
        Freezes a chromosome into static Python source.
        When absent, ResultSaver automatically falls back to dynamic mode.

Save modes (selected automatically by ResultSaver; train.py needs no changes)
------------------------------------------------------------------------------
    *static*  (arch implements ``generate_code``):
        Writes ``net.py`` + ``__init__.py`` (re-exports ``Net``).
        ``Net`` signature: ``Net(dropout, drop_connect_rate)``.
        Advantage: no runtime dependency on the archs package.

    *dynamic* (arch does not implement ``generate_code``):
        Saves ``chromosome.json`` and generates a wrapper ``__init__.py``.
        The wrapper subclasses ``arch.Net`` and injects the frozen
        ``individual`` / ``dataset`` at construction time.
        External ``Net`` signature is the same: ``Net(dropout, drop_connect_rate)``.
        Advantage: no codegen maintenance; always uses the latest decode logic.
"""

from __future__ import annotations

import importlib
import os
import types

_REQUIRED_ATTRS = ('Net', 'initialize_population', 'get_search_space')
_OPTIONAL_ATTRS = ('generate_code',)


def load_arch(arch_name: str) -> types.ModuleType:
    """
    Dynamically load ``archs/<arch_name>`` and validate its required interface.

    Raises
    ------
    ValueError
        If no subdirectory named *arch_name* exists under ``archs/``.
    TypeError
        If the arch package is missing one or more required attributes.
    """
    archs_dir = os.path.dirname(__file__)

    if not os.path.isdir(os.path.join(archs_dir, arch_name)):
        available = sorted(
            d for d in os.listdir(archs_dir)
            if os.path.isdir(os.path.join(archs_dir, d)) and not d.startswith('_')
        )
        raise ValueError(
            f"Unknown arch: {arch_name!r}. "
            f"Available (subdirs under archs/): {available}"
        )

    module = importlib.import_module(f'archs.{arch_name}')

    missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise TypeError(
            f"archs.{arch_name} is missing required attributes: {missing}.\n"
            f"Each arch's __init__.py must expose: {list(_REQUIRED_ATTRS)}"
        )

    return module


def has_codegen(arch_module: types.ModuleType) -> bool:
    """Return True if the arch implements the optional ``generate_code`` interface."""
    return hasattr(arch_module, 'generate_code') and callable(arch_module.generate_code)
