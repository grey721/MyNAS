"""
example_arch — ImageNet search space.

NETWORK TOPOLOGY
----------------
Follows the standard EfficientNet-family seven-stage layout::

    Input (3 × 224 × 224)
      │
    Stem  stride=2  3→32          (112×112)
      │
    Stage 1  ×1 block  stride=1  base 16   (112×112)  lightweight entry
    Stage 2  ×2 blocks stride=2  base 24   ( 56×56)
    Stage 3  ×2 blocks stride=2  base 40   ( 28×28)
    Stage 4  ×3 blocks stride=2  base 80   ( 14×14)
    Stage 5  ×3 blocks stride=1  base 112  ( 14×14)   capacity stage
    Stage 6  ×4 blocks stride=2  base 192  (  7×7 )   deep wide stage
    Stage 7  ×1 block  stride=1  base 320  (  7×7 )   final wide block
      │
    Head ConvBnAct(→1280) → AdaptiveAvgPool → Flatten
      │
    Dropout → Linear(1280 → 1000)

SEARCH-SPACE ENTRY LAYOUT
--------------------------
Identical to the CIFAR layout::

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
* Stage-1 entry: ``e`` fixed at ``{1}`` — MBConv1 at highest resolution.
* Stage-2/3 entries: ``e ∈ {3, 6}``; kernel limited to ``{3, 5}`` for speed.
* Stage-4 onwards: full kernel search ``{3, 5, 7}`` once spatial size is small.
* Stage-6 entry: ``e`` fixed at ``{6}`` — maximum capacity across the
  stride-2 boundary at the deepest stage.
* Stage-7: single wide block; ``f`` constrained to ``{1.0, 1.25}`` to
  guarantee the head receives sufficient channels.
"""

# fmt: off
search_space_imagenet: dict = {
    "names": [
        "stage_1_1",
        "stage_2_1", "stage_2_2",
        "stage_3_1", "stage_3_2",
        "stage_4_1", "stage_4_2", "stage_4_3",
        "stage_5_1", "stage_5_2", "stage_5_3",
        "stage_6_1", "stage_6_2", "stage_6_3", "stage_6_4",
        "stage_7_1",
    ],

    # [e_cands,   k_cands,    se_cands,          f_cands,              [base_ch], stride]

    # ── Stage 1 ── stride-1, 112×112, lightweight MBConv1 ────────────────
    "stage_1_1": [[1],        [3],       [0.0, 0.25], [0.75, 1.0, 1.25], [16],  1],

    # ── Stage 2 ── stride-2, 112→56 ──────────────────────────────────────
    "stage_2_1": [[3, 6],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [24],  2],
    "stage_2_2": [[1, 3, 6],  [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [24],  1],

    # ── Stage 3 ── stride-2, 56→28 ───────────────────────────────────────
    "stage_3_1": [[3, 6],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [40],  2],
    "stage_3_2": [[1, 3, 6],  [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [40],  1],

    # ── Stage 4 ── stride-2, 28→14, full kernel search ───────────────────
    "stage_4_1": [[3, 6],     [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [80],  2],
    "stage_4_2": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [80],  1],
    "stage_4_3": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [80],  1],

    # ── Stage 5 ── stride-1, 14×14, capacity stage ───────────────────────
    "stage_5_1": [[3, 6],     [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [112], 1],
    "stage_5_2": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [112], 1],
    "stage_5_3": [[1, 3, 6],  [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [112], 1],

    # ── Stage 6 ── stride-2, 14→7, deep wide stage ───────────────────────
    "stage_6_1": [[6],        [3, 5],    [0.0, 0.25], [0.75, 1.0, 1.25], [192], 2],
    "stage_6_2": [[3, 6],     [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [192], 1],
    "stage_6_3": [[3, 6],     [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [192], 1],
    "stage_6_4": [[3, 6],     [3, 5, 7], [0.0, 0.25], [0.75, 1.0, 1.25], [192], 1],

    # ── Stage 7 ── stride-1, 7×7, final wide block ───────────────────────
    "stage_7_1": [[6],        [3, 5],    [0.0, 0.25], [1.0,  1.25],       [320], 1],
}
# fmt: on
