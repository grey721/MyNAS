"""
example_arch genotypes sub-package.

Exports
-------
``get_search_space(dataset)``
    The *only* symbol the rest of the codebase should import from this
    sub-package.  Direct imports of ``search_space_cifar`` / etc. are
    intentionally avoided elsewhere to keep the dataset-split internal.

Extending to a new dataset
--------------------------
1. Create ``genotypes/<dataset>.py`` and define ``search_space_<dataset>``
   following the layout documented in ``cifar.py``.
2. Register the key(s) in ``_SEARCH_SPACES`` below.
3. No other file needs to change.
"""

from __future__ import annotations

from archs.example_arch.genotypes.cifar    import search_space_cifar
from archs.example_arch.genotypes.imagenet import search_space_imagenet

# Maps lower-cased dataset name → search-space dict.
# Multiple names may share the same dict (e.g. cifar10 / cifar100).
_SEARCH_SPACES: dict[str, dict] = {
    "cifar10":  search_space_cifar,
    "cifar100": search_space_cifar,
    "imagenet": search_space_imagenet,
}


def get_search_space(dataset: str) -> dict:
    """
    Return the search-space dictionary for *dataset*.

    Parameters
    ----------
    dataset : str
        Dataset name.  Case-insensitive.  Must be one of the keys in
        ``_SEARCH_SPACES`` (``'cifar10'``, ``'cifar100'``, ``'imagenet'``).

    Returns
    -------
    dict
        Contains key ``'names'`` (ordered block-name list) plus one entry
        per block mapping block_name → candidate-value lists.  See each
        ``genotypes/<dataset>.py`` for the exact per-entry layout.

    Raises
    ------
    ValueError
        If *dataset* is not supported.
    """
    key = dataset.lower()
    if key not in _SEARCH_SPACES:
        raise ValueError(
            f"example_arch does not support dataset {dataset!r}.  "
            f"Supported: {sorted(_SEARCH_SPACES)}"
        )
    return _SEARCH_SPACES[key]


__all__ = ["get_search_space"]
