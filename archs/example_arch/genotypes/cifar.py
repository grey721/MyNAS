"""
example_arch — CIFAR-10 / CIFAR-100 search space.

NETWORK TOPOLOGY
----------------
::

    Input (3 × 32 × 32)
      │
    Stem (stride=1, preserves 32×32 for small inputs)
      │
    Stage 1 ×4 blocks  stride=1  channels 32→48
      │
    Stage 2 ×4 blocks  stride=2  channels 64→80  (16×16)
      │
    Stage 3 ×4 blocks  stride=2  channels 96→128 (8×8)
      │
    Head → Classifier

CHANNEL PROGRESSION
-------------------
The first block of each stage uses a base channel that is wider than the
previous stage's output, creating a natural widening schedule.  Within a
stage the base channel grows gradually to allow the EA to modulate depth
by choosing the width multiplier ``f``.

SEARCH-SPACE ENTRY LAYOUT
--------------------------
::

    search_space[block_name] = [
        e_cands,    # list[int]   expansion-ratio candidates      (gene idx 0)
        k_cands,    # list[int]   depthwise kernel-size candidates (gene idx 1)
        se_cands,   # list[float] SE squeeze-ratio candidates      (gene idx 2)
        f_cands,    # list[float] channel scale-factor candidates  (gene idx 3)
        [base_ch],  # list[int]   singleton base output channels   (not sampled)
        stride,     # int         spatial stride                   (not sampled)
    ]

DESIGN NOTES
------------
* Stage-1 entry block (``stage_1_1``): expansion limited to ``{1, 3}``
  to keep the network lightweight near the highest spatial resolution.
* Stage-2 / Stage-3 entry blocks: expansion limited to ``{3, 6}``
  (stride-2, no identity shortcut — capacity needed across the stride).
* Last block of each stage (``stage_*_4``): ``f`` constrained to
  ``{1.0, 1.25}`` so the stage output never shrinks below its base width.
* All other blocks: full search over ``e ∈ {1,3,6}``, ``k ∈ {3,5,7}``,
  ``se ∈ {0.0, 0.25}``, ``f ∈ {0.75, 1.0, 1.25}``.
"""

# fmt: off
search_space_cifar: dict = {
    "names": [
        # Stage 1 — stride-1, spatial 32×32, channels 32→48
        "stage_1_1", "stage_1_2", "stage_1_3", "stage_1_4",
        # Stage 2 — stride-2 entry, spatial 32→16, channels 64→80
        "stage_2_1", "stage_2_2", "stage_2_3", "stage_2_4",
        # Stage 3 — stride-2 entry, spatial 16→8, channels 96→128
        "stage_3_1", "stage_3_2", "stage_3_3", "stage_3_4",
    ],

    # ── Stage 1 ───────────────────────────────────────────────────────────
    # [e_cands,   k_cands,    se_cands,          f_cands,              [base_ch], stride]
    "stage_1_1": [[1, 3],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [32],  1],
    "stage_1_2": [[1, 3, 6],  [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [40],  1],
    "stage_1_3": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [48],  1],
    "stage_1_4": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [1.0,  1.25],       [48],  1],

    # ── Stage 2 ───────────────────────────────────────────────────────────
    "stage_2_1": [[3, 6],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [64],  2],
    "stage_2_2": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [72],  1],
    "stage_2_3": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [80],  1],
    "stage_2_4": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [1.0,  1.25],       [80],  1],

    # ── Stage 3 ───────────────────────────────────────────────────────────
    "stage_3_1": [[3, 6],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [96],  2],
    "stage_3_2": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [112], 1],
    "stage_3_3": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [128], 1],
    "stage_3_4": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [1.0,  1.25],       [128], 1],
}
# fmt: on
