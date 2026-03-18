# MyNAS — Evolutionary Neural Architecture Search

A training-free, multi-objective evolutionary NAS framework built on
zero-cost proxies.  The search phase runs entirely without gradient-based
training, making it practical on a single GPU in minutes rather than days.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [1. Architecture Search](#1-architecture-search)
  - [2. Training the Found Architecture](#2-training-the-found-architecture)
- [Search Arguments](#search-arguments)
- [Training Arguments](#training-arguments)
- [Training Outputs](#training-outputs)
- [Zero-Cost Proxies](#zero-cost-proxies)
  - [Built-in Proxies](#built-in-proxies)
  - [Adding a Custom Proxy](#adding-a-custom-proxy)
- [Adding a New Architecture](#adding-a-new-architecture)
- [Search Output Files](#search-output-files)
- [Design Notes](#design-notes)

---

## Overview

MyNAS decouples the search phase from the training phase:

```
Search  (zero-cost, seconds per generation)
  ┌──────────────────────────────────────────────────────────────┐
  │  Population of chromosomes  ──►  Zero-Cost Proxy score       │
  │  (random init / crossover / mutation)         │              │
  │                                               ▼              │
  │  Dual-population NSGA-II  (P1: constraint-elite pool)        │
  │                           (P2: Pareto-diversity pool)        │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼  best chromosome  →  scripts/<n>/
Training  (standard SGD, fully independent of search)
  ┌──────────────────────────────────────────────────────────────┐
  │  train.py loads Net from scripts/<n>/__init__.py             │
  │  Cosine-annealing LR  +  linear warmup  +  DropPath          │
  │  Interrupt-safe: data and curves saved even on early exit    │
  └──────────────────────────────────────────────────────────────┘
```

Key design decisions:

- **Direct-import proxies** — no registry, no magic; import whichever proxy
  class you need and pass an instance to `Evaluator` or `Searcher`.
- **Arch plugin system** — new architectures live in `archs/<n>/` and expose a
  uniform three-symbol interface; `ga.py` has zero knowledge of gene structure.
- **Static / dynamic save modes** — if an arch implements `generate_code()`,
  the found architecture is frozen to a self-contained `net.py`; otherwise a
  lightweight wrapper decodes the chromosome at runtime.
- **Interrupt-safe training** — the training loop uses `try/except/finally`;
  CSV, JSON, and PNG outputs are always written to disk even when training is
  cut short by `Ctrl-C` or an unexpected error.

---

## Project Structure

```
MyNAS/
├── search.py                     Entry point for architecture search
├── train.py                      Entry point for network training
│
├── archs/                        Architecture definitions (plug-in system)
│   ├── __init__.py               load_arch() factory + has_codegen() helper
│   │
│   ├── reference_arch/           Canonical reference implementation (start here)
│   │   ├── __init__.py           Exposes all required + optional interfaces;
│   │   │                         full documentation of every interface contract
│   │   ├── genotypes/
│   │   │   ├── __init__.py       get_search_space(dataset) dispatcher
│   │   │   ├── cifar.py          CIFAR-10/100 search space  (12 blocks × 4 genes)
│   │   │   └── imagenet.py       ImageNet search space      (16 blocks × 4 genes)
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   └── ops.py            make_divisible / ConvBnAct / SqueezeExcitation /
│   │   │                         MBConvBlock
│   │   └── net/
│   │       ├── __init__.py
│   │       ├── net.py            Dynamic Net  (chromosome → nn.Module at runtime)
│   │       ├── population_initializer.py  initialize_population + sample_gene
│   │       └── codegen.py        generate_code  (chromosome → static net.py)
│   │
│   ├── lightmix_arch/            LightMix architecture (ported from paper)
│   │   ├── __init__.py
│   │   ├── genotypes/
│   │   │   ├── __init__.py       get_search_space(dataset) dispatcher
│   │   │   ├── cifar.py          CIFAR search space  (30 blocks × 5 genes)
│   │   │   └── imagenet.py       ImageNet search space
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   └── block.py          LightMix operators: Hswish / SELayer / ECALayer /
│   │   │                         MixedScaleModule / MDConv / GroupedConv2d / Block
│   │   └── net/
│   │       ├── __init__.py
│   │       ├── net.py            Dynamic Net
│   │       ├── population_initializer.py
│   │       └── codegen.py
│   │
│   └── example_arch/             Minimal stub (kept for reference; prefer reference_arch)
│       └── ...
│
├── ea/                           Evolutionary algorithm
│   ├── ga.py                     Searcher — crossover / mutation / selection
│   ├── evaluate.py               Evaluator — population fitness scoring
│   ├── select.py                 non_dominated_sort + crowding_distance
│   └── proxy/                    Zero-cost proxy subsystem
│       ├── __init__.py           CompositeProxy + public API
│       ├── base.py               BaseProxy ABC
│       ├── naswot.py             NAS-WOT proxy
│       └── synflow.py            SynFlow proxy
│
├── load_dataset/                 Data loading utilities
│   ├── loaders.py                get_train / test / nas / debug loader + AugLevel
│   ├── autoaugment.py            CIFAR10Policy / ImageNetPolicy
│   └── random_erasing.py         RandomErasing transform
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 Logger — file + console logging, training-history
│   │                             persistence (CSV + JSON), curve plotting (PNG)
│   └── saver.py                  plot_population / save_population_info / ResultSaver
│
├── template/                     Reusable building blocks (arch-independent)
│   ├── func.py                   init_weight / Conv / SE / EfficientNet / EMA / losses
│   ├── drop.py                   DropPath / DropBlock  (Ross Wightman, Apache 2.0)
│   └── tools.py                  cal_flops_params
│
├── infotool/                     FLOPs / parameter counting backend
│
├── scripts/                      Generated by search.py; one sub-directory per result
│   └── <arch_name>/
│       ├── __init__.py           Net entry point (static re-export or dynamic wrapper)
│       ├── net.py                (static mode only) frozen network definition
│       ├── chromosome.json       Raw chromosome array
│       ├── search_info.json      Search metadata (proxy, objectives, save mode …)
│       ├── train.sh              Ready-to-run training script
│       └── README.txt            Human-readable result summary
│
├── logs/                         Experiment logs (auto-created at runtime)
│   └── Train_<n>_<timestamp>/
│       ├── run.log               Full console + file log
│       ├── config.json           Training hyper-parameters
│       ├── training_history.csv  Per-epoch metrics (epoch/loss/acc/lr)
│       ├── training_summary.json Scalar summary (best_acc, best_epoch …)
│       └── plots/
│           └── training_curves.png  3-panel figure (Loss / Acc / LR)
│
└── trained_models/               Saved checkpoints (auto-created at runtime)
    └── <n>_best.pt               Best test-accuracy checkpoint
```

---

## Requirements

```
Python     >= 3.10
PyTorch    >= 1.10
torchvision
numpy
pandas
matplotlib
```

Install:

```bash
pip install torch torchvision numpy pandas matplotlib
```

Datasets are expected under `../datasets/` by default (see
`load_dataset/loaders.py` → `DEFAULT_DATA_ROOTS`).  Override by editing that
dict or by symlinking.

---

## Quick Start

### 1. Architecture Search

```bash
python search.py \
    --arch              reference_arch \
    --dataset           cifar10 \
    --proxy             naswot \
    --p1_size           50 \
    --p2_size           50 \
    --num_generations   50 \
    --batch_size_search 128 \
    --params_max        0.5 \
    --file_name         Best_architecture
```

Results are written to `scripts/Best_architecture/`.

#### Search with a different proxy

```bash
python search.py --arch reference_arch --dataset cifar100 --proxy synflow
```

#### Unconstrained search

Omit `--params_max` / `--flops_max` to obtain an unconstrained Pareto front.

---

### 2. Training the Found Architecture

The simplest way is to run the generated shell script:

```bash
bash scripts/Best_architecture/train.sh
```

Or invoke `train.py` directly:

```bash
python train.py \
    --script_name        Best_architecture \
    --dataset            cifar10 \
    --total_epochs       600 \
    --warmup_epochs      10 \
    --lr                 0.05 \
    --weight_decay       4e-5 \
    --aug_level          basic \
    --dropout            0.2 \
    --drop_connect_rate  0.2
```

Training is **interrupt-safe**.  Pressing `Ctrl-C` or an unexpected crash will
trigger a clean save of all data collected up to that point before the process
exits.  See [Training Outputs](#training-outputs) for details.

---

## Search Arguments

| Argument | Default | Description |
|---|---|---|
| `--arch` | *(required)* | Arch name matching `archs/<arch>/` directory |
| `--dataset` | *(required)* | `cifar10` / `cifar100` / `imagenet` |
| `--proxy` | `naswot` | Zero-cost proxy: `naswot` / `synflow` |
| `--p1_size` | `40` | P1 constraint-elite pool size |
| `--p2_size` | `40` | P2 Pareto-diversity pool size |
| `--num_generations` | `50` | Number of evolutionary generations |
| `--crossover_rate` | `0.8` | Crossover probability |
| `--mutation_rate` | `0.2` | Per-gene mutation probability |
| `--batch_size_search` | `128` | Batch size for proxy evaluation |
| `--params_max` | `None` | Parameter count upper bound (M) |
| `--params_min` | `None` | Parameter count lower bound (M) |
| `--flops_max` | `None` | FLOPs upper bound (M) |
| `--flops_min` | `None` | FLOPs lower bound (M) |
| `--random_seed` | `42` | Global random seed |
| `--file_name` | `Best_architecture` | Output script name |

---

## Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--script_name` | *(required)* | Sub-directory under `scripts/` to load |
| `--dataset` | *(required)* | Dataset name |
| `--gpu` | `0` | GPU device index |
| `--batch_size` | `128` | Training batch size |
| `--total_epochs` | `600` | Total training epochs |
| `--warmup_epochs` | `10` | Linear LR warmup epochs |
| `--lr` | `0.05` | Peak learning rate |
| `--warmup_lr` | `0.0001` | Initial LR during warmup |
| `--weight_decay` | `4e-5` | SGD weight decay |
| `--momentum` | `0.9` | SGD momentum |
| `--grad_clip` | `5.0` | Gradient norm clip threshold (`0` = disabled) |
| `--dropout` | `0.2` | Classifier head dropout |
| `--drop_connect_rate` | `0.2` | DropPath ceiling (linearly scaled per epoch) |
| `--aug_level` | `basic` | Augmentation: `none` / `basic` / `strong` |
| `--use_subprocess` | `False` | Run in a child process (releases GPU on exit) |
| `--seed` | `42` | Random seed |

---

## Training Outputs

All training outputs are written **once, at the end of training** — including
when training is interrupted or crashes.  The training loop is structured as:

```
try:
    for epoch in range(total_epochs):
        train + validate

except KeyboardInterrupt:
    log warning, do not re-raise      ← Ctrl-C is treated as clean partial finish

except Exception:
    log error, then re-raise          ← propagate; finally still runs first

finally:
    _save_outputs()                   ← always executes
        ├── write training_history.csv  +  training_summary.json
        └── render training_curves.png
```

Each save step runs in its own `try/except`, so a failure in one step (e.g.
matplotlib out-of-memory) cannot prevent the others (e.g. CSV write) from
completing.

### Output files

#### `trained_models/<n>_best.pt`

Model state dict from the epoch with the highest test accuracy.  Written
in-place by `_validate()` whenever a new best is found during training.

#### `logs/<run>/training_history.csv`

Per-epoch metrics table, designed to be loaded directly into pandas or numpy
for drawing comparison and ablation experiment curves.

| Column | Type | Description |
|---|---|---|
| `epoch` | int | 1-based epoch index |
| `train_loss` | float | Mean cross-entropy on the training set |
| `test_loss` | float | Mean cross-entropy on the test set |
| `train_acc` | float | Training accuracy ∈ [0, 1] |
| `test_acc` | float | Test accuracy ∈ [0, 1] |
| `lr` | float | Learning rate at the start of the epoch |

For interrupted runs the table contains only the rows for completed epochs;
`len(df)` gives the actual number of epochs run.

**Example — overlay two runs for a comparison plot:**

```python
import pandas as pd
import matplotlib.pyplot as plt

df_a = pd.read_csv('logs/Train_ModelA_.../training_history.csv')
df_b = pd.read_csv('logs/Train_ModelB_.../training_history.csv')

fig, ax = plt.subplots()
ax.plot(df_a['epoch'], df_a['test_acc'] * 100, label='Model A')
ax.plot(df_b['epoch'], df_b['test_acc'] * 100, label='Model B')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Acc (%)')
ax.legend()
plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
```

#### `logs/<run>/training_summary.json`

Scalar summary for quick inspection.

| Field | Description |
|---|---|
| `best_acc` | Highest test accuracy observed |
| `best_acc_epoch` | Epoch at which `best_acc` was achieved |
| `best_loss` | Lowest test loss observed |
| `best_loss_epoch` | Epoch at which `best_loss` was achieved |
| `final_train_acc` | Training accuracy at the last completed epoch |
| `final_test_acc` | Test accuracy at the last completed epoch |
| `final_lr` | Learning rate at the last completed epoch |

#### `logs/<run>/plots/training_curves.png`

Three-panel figure saved at the end of training (or on interruption):

```
[ Loss ]  [ Accuracy ]  [ Learning Rate ]
```

The best point on each applicable curve is marked with a star annotation
showing the exact value.

---

## Zero-Cost Proxies

Proxies are plain Python classes.  Import whichever class you need and pass an
instance directly — there is no string-based registry.

### Built-in Proxies

| Class | Module | Paper | Score signal |
|---|---|---|---|
| `NASWOT` | `ea/proxy/naswot.py` | Mellor et al., ICML 2021 | log\|K\|: log-determinant of ReLU activation kernel |
| `SynFlow` | `ea/proxy/synflow.py` | Tanaka et al., NeurIPS 2020 | Σ\|θ ⊙ ∂R/∂θ\|, data-free synaptic flow |

### Usage

**Single proxy:**

```python
from ea.proxy.naswot  import NASWOT
from ea.proxy.synflow import SynFlow
from ea.evaluate      import Evaluator

evaluator = Evaluator('reference_arch', 'cifar10', batch_size=128,
                      proxy=NASWOT(batch_size=128))
# or
evaluator = Evaluator('reference_arch', 'cifar10', batch_size=128,
                      proxy=SynFlow())
```

**Composite proxy (weighted sum):**

```python
from ea.proxy         import CompositeProxy
from ea.proxy.naswot  import NASWOT
from ea.proxy.synflow import SynFlow

proxy = CompositeProxy(
    proxies=[NASWOT(batch_size=128), SynFlow()],
    weights=[0.7, 0.3],   # auto-normalised; equal weights if omitted
)
evaluator = Evaluator('reference_arch', 'cifar10', batch_size=128, proxy=proxy)
```

**From the CLI:**

```bash
python search.py --arch reference_arch --dataset cifar10 --proxy naswot
python search.py --arch reference_arch --dataset cifar10 --proxy synflow
```

### Adding a Custom Proxy

**Step 1** — Create `ea/proxy/grad_norm.py`:

```python
import torch
import torch.nn as nn
from ea.proxy.base import BaseProxy

class GradNormProxy(BaseProxy):
    """Sum of squared parameter gradients (higher = better)."""

    def _compute(
        self,
        net: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        x, target = batch
        net.zero_grad()
        loss = nn.CrossEntropyLoss()(net(x), target)
        loss.backward()
        score = sum(
            p.grad.norm().item() ** 2
            for p in net.parameters() if p.grad is not None
        )
        net.zero_grad()
        return score
```

**Step 2** — Use it directly:

```python
from ea.proxy.grad_norm import GradNormProxy
from ea.evaluate        import Evaluator

evaluator = Evaluator('reference_arch', 'cifar10', batch_size=128,
                      proxy=GradNormProxy())
```

**Step 3 (optional)** — Support `--proxy grad_norm` from the CLI by adding
one line to `_build_proxy()` in `search.py`:

```python
_PROXY_MAP = {
    'naswot':    NASWOT(batch_size=batch_size),
    'synflow':   SynFlow(),
    'grad_norm': GradNormProxy(),   # ← add this line
}
```

No other files need to change.

---

## Adding a New Architecture

The recommended starting point is `archs/reference_arch/`, which is a
complete, production-quality reference implementation with full documentation
of every interface contract.  Copy it, rename it, and adapt each file in turn.

`archs/load_arch()` validates that every arch exposes the three required
symbols from its `__init__.py`.  Missing any of them raises `TypeError` at
startup.

### Required interface

| Symbol | Signature | Called by |
|---|---|---|
| `Net` | `Net(individual, dataset, dropout=0.0, drop_connect_rate=0.0) → nn.Module` | `Evaluator.evaluate()` |
| `initialize_population` | `(search_space, pop_size) → ndarray  shape (pop_size, num_blocks, G)` | `Searcher._initialize_population()` |
| `get_search_space` | `(dataset: str) → dict` | `Searcher.__init__()`, `Evaluator.__init__()` |

### Optional interface

| Symbol | Signature | Effect when present |
|---|---|---|
| `generate_code` | `(individual, dataset) → str` | Enables **static save mode**: search result contains a frozen `net.py` with no runtime dependency on the `archs` package. |
| `sample_gene` | `(search_space, block_name) → list` | Used by `Searcher.mute()` for per-block mutation.  Without it, `mute()` falls back to uniform sampling over each candidate list, which may violate arch-specific structural constraints. |

### Search-space dict format

```python
{
    'names': ['block_0', 'block_1', ...],   # ordered block-name list

    'block_0': [
        [cand_0_0, cand_0_1, ...],   # gene 0 candidates  (e.g. expansion ratios)
        [cand_1_0, cand_1_1, ...],   # gene 1 candidates  (e.g. kernel sizes)
        ...                           # one list per sampled gene  (indices 0 … G-1)
        [base_ch],                    # singleton: base output channel count
        stride,                       # int scalar: spatial stride (1 or 2)
    ],
    ...
}
```

Indices `0 … G-1` are sampled by `initialize_population` and `sample_gene`.
The base-channel entry is a singleton list accessed as
`search_space[name][-2][0]`.  The stride is read directly as
`search_space[name][-1]`.

### Static vs dynamic save mode

When `generate_code` is absent, `ResultSaver` automatically uses *dynamic
mode*: `chromosome.json` is saved and the generated `__init__.py` wraps
`arch.Net`, injecting the frozen chromosome at construction time.  Both modes
expose the identical `Net(dropout, drop_connect_rate)` signature to `train.py`.

| | Static | Dynamic |
|---|---|---|
| Requires `generate_code` | Yes | No |
| `net.py` written to `scripts/` | Yes | No |
| Runtime dep on `archs/` pkg at train time | No | Yes |
| Decode logic always up-to-date | No (frozen at search) | Yes |

---

## Search Output Files

After a successful search, `scripts/<file_name>/` contains:

```
scripts/<file_name>/
├── __init__.py        Net entry point (identical external interface in both modes)
├── chromosome.json    Raw chromosome array
├── search_info.json   Search metadata
├── train.sh           Ready-to-run training script
├── README.txt         Human-readable result summary
└── net.py             (static mode only) frozen network definition
```

`search_info.json` example:

```json
{
    "name": "Best_architecture",
    "arch": "reference_arch",
    "dataset": "cifar10",
    "net_mode": "static",
    "saved_at": "2025-01-01 12:00:00",
    "objectives": {
        "error":    0.0523,
        "params_M": 0.48,
        "flops_M":  182.4
    },
    "proxy": "NasWotProxy(batch_size=128)",
    "generations": 100,
    "p1_size": 50,
    "p2_size": 50
}
```

---

## Design Notes

### Dual-population selection

The searcher maintains two independent pools:

- **P1** (constraint-elite) — feasibility-first: individuals satisfying all
  constraints are selected by non-dominated rank, then by fitness.  Infeasible
  individuals fill remaining slots ordered by constraint-violation penalty.
- **P2** (Pareto-diversity) — standard NSGA-II: non-dominated sort + crowding
  distance, no explicit constraint handling.

Both pools exchange offspring each generation, allowing P1 to exploit
constraint-feasible regions while P2 maintains diversity across the full
Pareto front.

### Interrupt-safe training

See [Training Outputs](#training-outputs) for the full `try/except/finally`
structure.  The key invariant is: **any run that completes at least one epoch
will produce a CSV, a JSON summary, and a PNG figure**, regardless of how it
ends.

### Score convention

All proxies return **higher-is-better** floats.  `Evaluator` stores the
negated value in column `COL_ERR` for multi-objective selection.  Never
negate a proxy score twice.

All proxies return **higher-is-better** floats.  `Evaluator` stores the
negated value in column `COL_ERR` for multi-objective selection.  Never
negate a proxy score twice.
