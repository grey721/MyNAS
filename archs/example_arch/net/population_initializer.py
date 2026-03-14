"""
example_arch population initializer.

Migrated from ``ea/ga.py``'s ``_initialize_population``; now part of the arch
so that ``ga.py`` has zero knowledge of the concrete parameter structure.

Gene encoding  [s, e, b, k, se, f]
-----------------------------------
  idx 0  s   : float  Split ratio.
  idx 1  e   : int    Expansion rate.
  idx 2  b   : int    Branch combination bitmask.
  idx 3  k   : int    Depthwise kernel size.
  idx 4  se  : float  Squeeze-and-Excitation ratio.
  idx 5  f   : float  Channel scale factor.

Handling s=0 (skip block)
--------------------------
  When 0 is in the candidate list for ``s``, ``s=0`` means "skip this block"
  (identity). To keep the population depth distribution well-spread:

  - Each individual is assigned a random ``rand_len`` value as a relative
    "expected network depth" threshold.
  - Block at position ``j`` is skipped (s=0) when its relative depth
    ``j / num_blocks`` is large relative to ``rand_len``.

  This produces a realistic distribution where early blocks are more likely
  to be kept and deeper blocks are more likely to be skipped.
"""

from __future__ import annotations

import numpy as np

# Number of gene slots per block — must match the candidate lists in cifar.py.
NUM_GENE_PER_BLOCK = 6  # [s, e, b, k, se, f]


def initialize_population(search_space: dict, pop_size: int) -> np.ndarray:
    """
    Randomly initialise a population of chromosomes.
    """
    names = search_space['names']
    num_blocks = len(names)

    population = np.empty((pop_size, num_blocks, NUM_GENE_PER_BLOCK), dtype=object)

    # Per-individual random depth threshold for controlling s=0 probability.
    rand_len = np.random.rand(pop_size)  # (pop_size,)
    rand_s = np.random.rand(pop_size, num_blocks)  # (pop_size, num_blocks)

    for j, name in enumerate(names):
        s_cands, e_cands, b_cands, k_cands, se_cands, f_cands = search_space[name][:6]

        # --- Special sampling for s ---
        if 0 in s_cands:
            s_nonzero = [x for x in s_cands if x != 0]
            # Deeper blocks are more likely to be skipped.
            rel_pos = j / num_blocks
            skip_mask = rand_s[:, j] > (1.0 - rand_len * (1.0 - rel_pos))
            s_values = np.zeros(pop_size, dtype=object)
            keep_mask = ~skip_mask
            if keep_mask.any():
                s_values[keep_mask] = np.random.choice(s_nonzero, size=keep_mask.sum())
        else:
            s_values = np.array(
                np.random.choice(s_cands, size=pop_size), dtype=object
            )

        population[:, j, 0] = s_values
        population[:, j, 1] = np.random.choice(e_cands, size=pop_size).astype(object)
        population[:, j, 2] = np.random.choice(b_cands, size=pop_size).astype(object)
        population[:, j, 3] = np.random.choice(k_cands, size=pop_size).astype(object)
        population[:, j, 4] = np.random.choice(se_cands, size=pop_size).astype(object)
        population[:, j, 5] = np.random.choice(f_cands, size=pop_size).astype(object)

    return population


def sample_gene(search_space: dict, block_name: str) -> list:
    """
    Sample a single gene vector for one block at random.

    Used by the mutation operator in ``ga.py``.

    Returns
    -------
    list of length ``NUM_GENE_PER_BLOCK``: ``[s, e, b, k, se, f]``.
    """
    s_cands, e_cands, b_cands, k_cands, se_cands, f_cands = search_space[block_name][:6]
    return [
        float(np.random.choice(s_cands)),
        int(np.random.choice(e_cands)),
        int(np.random.choice(b_cands)),
        int(np.random.choice(k_cands)),
        float(np.random.choice(se_cands)),
        float(np.random.choice(f_cands)),
    ]


__all__ = ['initialize_population', 'sample_gene', 'NUM_GENE_PER_BLOCK']
