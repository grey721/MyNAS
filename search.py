import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ea.evaluate import COL_PARAMS, COL_FLOPS
from ea.ga import Searcher
from ea.proxy.base import BaseProxy
from ea.proxy.naswot import NASWOT
from ea.proxy.synflow import SynFlow


# ---------------------------------------------------------------------------
# CLI name -> proxy factory
#
# This is the only place in the codebase that maps a user-supplied string to
# a concrete proxy class.  It is intentionally kept small and local: it is
# application-level glue, not a registry.  When a new proxy is added, one
# line is added here.
# ---------------------------------------------------------------------------
def _build_proxy(name: str, batch_size: int) -> BaseProxy:
    """
    Construct a proxy instance from a CLI name string.

    Parameters
    ----------
    name:
        One of the supported proxy names (case-insensitive).
    batch_size:
        Passed to proxies that require it (e.g. ``NasWotProxy``).

    Raises
    ------
    ValueError
        If *name* does not match any known proxy.
    """
    _PROXY_MAP: dict[str, BaseProxy] = {
        'naswot': NASWOT(batch_size=batch_size),
        'synflow': SynFlow(),
    }
    key = name.lower()
    if key not in _PROXY_MAP:
        raise ValueError(
            f"Unknown proxy: {name!r}. "
            f"Available: {sorted(_PROXY_MAP)}"
        )
    return _PROXY_MAP[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evolutionary Architecture Search')

    # --- Arch & proxy ---
    parser.add_argument(
        '--arch', type=str, required=True,
        help='Subdirectory name under archs/, e.g. example_arch. '
             'Must match the archs/<arch>/ directory name exactly.',
    )
    parser.add_argument(
        '--proxy', type=str, default='naswot',
        help='Zero-cost proxy name (default: naswot). '
             'Available: naswot, synflow.',
    )

    # --- Dataset & random seed ---
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=42)

    # --- Evolutionary hyperparameters ---
    parser.add_argument('--p1_size', type=int, default=50)
    parser.add_argument('--p2_size', type=int, default=50)
    parser.add_argument('--num_generations', type=int, default=50)
    parser.add_argument('--mutation_rate', type=float, default=0.2)
    parser.add_argument('--crossover_rate', type=float, default=0.8)
    parser.add_argument('--batch_size_search', type=int, default=128)

    # --- Constraints (unit: M; internally converted to raw values) ---
    parser.add_argument('--params_max', type=float, default=None,
                        help='Parameter count upper bound (M), e.g. 0.5')
    parser.add_argument('--params_min', type=float, default=None,
                        help='Parameter count lower bound (M)')
    parser.add_argument('--flops_max', type=float, default=None,
                        help='FLOPs upper bound (M), e.g. 200')
    parser.add_argument('--flops_min', type=float, default=None,
                        help='FLOPs lower bound (M)')

    # --- Output file name ---
    parser.add_argument('--file_name', type=str, default='Best_architecture',
                        help='Output script name (without extension)')

    return parser.parse_args()


def _build_constraints(args: argparse.Namespace) -> dict:
    """
    Convert CLI constraint arguments (unit: M) to the internal format
    (raw values, keyed by column index).

    Returns an empty dict when no constraints are specified.
    """
    raw: dict[int, tuple[Optional[float], Optional[float]]] = {
        COL_PARAMS: (args.params_min, args.params_max),
        COL_FLOPS: (args.flops_min, args.flops_max),
    }
    constraints = {}
    for col, (lo, hi) in raw.items():
        lo_v = lo * 1e6 if lo is not None else None
        hi_v = hi * 1e6 if hi is not None else None
        if lo_v is not None or hi_v is not None:
            constraints[col] = (lo_v, hi_v)
    return constraints


def _ensure_dirs() -> None:
    for d in ['./logs', './scripts', './trained_models']:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    args = parse_args()

    _ensure_dirs()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    constraints = _build_constraints(args)

    # Construct the proxy here; Searcher and Evaluator receive an instance.
    proxy = _build_proxy(args.proxy, batch_size=args.batch_size_search)

    searcher = Searcher(
        arch_name=args.arch,
        dataset=args.dataset,
        batch_size_search=args.batch_size_search,
        p1_size=args.p1_size,
        p2_size=args.p2_size,
        generations=args.num_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        constraints=constraints,
        proxy=proxy,
    )

    searcher.log.save_config(args)
    searcher.evolve(file_name=args.file_name)
