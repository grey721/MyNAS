"""
example_arch population initialiser.

RESPONSIBILITIES
----------------
``initialize_population``
    Creates the initial P1 / P2 populations at the start of search.
    Called by ``Searcher._initialize_population()`` in ``ga.py``.

``sample_gene``
    Samples a single gene vector for one named block.
    Called by ``Searcher.mute()`` for per-block mutation.
    By living in the arch, it lets ``ga.py`` stay fully decoupled from
    the concrete gene structure.

GENE ENCODING
-------------
Each gene is a vector of length ``NUM_GENE_PER_BLOCK = 4``::

    [e, k, se, f]

    idx 0  e  : int   Expansion ratio
    idx 1  k  : int   Depthwise kernel size
    idx 2  se : float SE squeeze ratio
    idx 3  f  : float Channel width multiplier

CHROMOSOME SHAPE
----------------
A chromosome is a 2-D object array of shape ``(num_blocks, 4)``.
The full population has shape ``(pop_size, num_blocks, 4)``.

``dtype=object`` is used throughout so that each cell can hold either an
``int`` or a ``float`` without silent promotion to ``float64``.  ``Net``
reads each cell with an explicit ``int(gene[i])`` / ``float(gene[i])`` cast.

VECTORISED SAMPLING
-------------------
All ``pop_size`` individuals are sampled for each block in a single
``np.random.choice`` call (one call per gene dimension per block).
This is significantly faster than nested Python loops for large populations
and avoids any Python-level per-individual branching.

EXTENDING THE GENE
------------------
If the gene encoding changes (e.g. adding a ``d`` depth-scaling gene),
update ``NUM_GENE_PER_BLOCK`` and the sampling lines in both functions.
The framework reads this constant from
``archs.example_arch.net.NUM_GENE_PER_BLOCK`` when it needs the gene
length (e.g. for shape assertions in tests).
"""

from __future__ import annotations

import numpy as np

# Must equal the number of sampled candidate-list fields (indices 0–3)
# in each search-space entry.  Update both here and in the search space
# if the gene encoding changes.
NUM_GENE_PER_BLOCK: int = 4   # [e, k, se, f]


def initialize_population(search_space: dict, pop_size: int) -> np.ndarray:
    """
    Randomly initialise a population of chromosomes.

    Each gene value is drawn uniformly at random from the corresponding
    candidate list.  No structural constraints beyond the candidate lists
    are enforced here; such constraints should be expressed by restricting
    the candidate lists in the search space (e.g. to prevent an expansion
    ratio of 1 on a stride-2 entry block, simply omit 1 from its e_cands).

    Parameters
    ----------
    search_space : dict
        Returned by ``get_search_space(dataset)``.
    pop_size : int
        Number of individuals to generate.

    Returns
    -------
    np.ndarray
        Shape ``(pop_size, num_blocks, NUM_GENE_PER_BLOCK)``, dtype=object.
    """
    names      = search_space["names"]
    num_blocks = len(names)

    population = np.empty((pop_size, num_blocks, NUM_GENE_PER_BLOCK), dtype=object)

    for j, name in enumerate(names):
        e_cands, k_cands, se_cands, f_cands = search_space[name][:4]

        # One vectorised call per gene dimension per block.
        population[:, j, 0] = np.random.choice(e_cands,  size=pop_size).astype(object)
        population[:, j, 1] = np.random.choice(k_cands,  size=pop_size).astype(object)
        population[:, j, 2] = np.random.choice(se_cands, size=pop_size).astype(object)
        population[:, j, 3] = np.random.choice(f_cands,  size=pop_size).astype(object)

    return population


def sample_gene(search_space: dict, block_name: str) -> list:
    """
    Sample a single gene vector for one block uniformly at random.

    This is the per-block mutation primitive consumed by ``Searcher.mute()``
    in ``ga.py``.  Keeping it inside the arch ensures ``ga.py`` has zero
    knowledge of the concrete gene structure.

    Parameters
    ----------
    search_space : dict
        Same object as passed to ``initialize_population``.
    block_name : str
        One of the entries in ``search_space['names']``.

    Returns
    -------
    list, length ``NUM_GENE_PER_BLOCK``
        ``[e, k, se, f]`` — types match what ``Net`` expects:
        ``e`` and ``k`` are ``int``; ``se`` and ``f`` are ``float``.

    Examples
    --------
    >>> from archs.example_arch.genotypes import get_search_space
    >>> from archs.example_arch.net.population_initializer import sample_gene
    >>> ss   = get_search_space('cifar10')
    >>> gene = sample_gene(ss, 'stage_1_1')
    >>> len(gene) == 4 and isinstance(gene[0], int)
    True
    """
    e_cands, k_cands, se_cands, f_cands = search_space[block_name][:4]
    return [
        int(np.random.choice(e_cands)),
        int(np.random.choice(k_cands)),
        float(np.random.choice(se_cands)),
        float(np.random.choice(f_cands)),
    ]


__all__ = ["initialize_population", "sample_gene", "NUM_GENE_PER_BLOCK"]
