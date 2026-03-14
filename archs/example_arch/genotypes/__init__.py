from archs.example_arch.genotypes.cifar import search_space_cifar

_SEARCH_SPACES = {
    'cifar10':  search_space_cifar,
    'cifar100': search_space_cifar,
    # 'imagenet': search_space_imagenet,  # TODO
}


def get_search_space(dataset: str) -> dict:
    """
    Return the search-space dictionary for the given dataset.

    Parameters
    ----------
    dataset:
        Dataset name, e.g. ``'cifar10'`` or ``'cifar100'``.

    Returns
    -------
    dict
        Contains a ``'names'`` key (ordered block list) plus one entry per block.

    Raises
    ------
    ValueError
        If *dataset* is not supported.
    """
    key = dataset.lower()
    if key not in _SEARCH_SPACES:
        raise ValueError(
            f"CSBConv does not support dataset: {dataset!r}. "
            f"Available: {list(_SEARCH_SPACES)}"
        )
    return _SEARCH_SPACES[key]


__all__ = ['get_search_space']
