"""
Shared utility functions and classes.

Contents
--------
    plot_population      : Scatter plots of a population in objective space
                           (called by Logger).
    save_population_info : Persist population data to CSV + JSON.
    ResultSaver          : Search result saver; auto-selects static / dynamic mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Population visualisation
# ---------------------------------------------------------------------------

_OBJ_LABELS = ['Error', 'Params (M)', 'FLOPs (M)']
_PROJECTIONS = [
    (0, 1, 'err-params'),
    (0, 2, 'err-flops'),
    (1, 2, 'params-flops'),
]


def plot_population(
        p1_fitness: np.ndarray,
        p2_fitness: np.ndarray,
        save_dir: str,
        generation: int,
) -> None:
    """Plot P1/P2 populations across all three 2-D objective projections."""
    os.makedirs(save_dir, exist_ok=True)

    p1 = p1_fitness[:, 1:].copy()
    p2 = p2_fitness[:, 1:].copy()
    for arr in (p1, p2):
        arr[:, 1] /= 1e6
        arr[:, 2] /= 1e6

    for x_idx, y_idx, suffix in _PROJECTIONS:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(p1[:, x_idx], p1[:, y_idx], c='steelblue', label='P1', alpha=0.7, s=20)
        ax.scatter(p2[:, x_idx], p2[:, y_idx], c='tomato', label='P2', alpha=0.7, s=20)
        ax.set_xlabel(_OBJ_LABELS[x_idx])
        ax.set_ylabel(_OBJ_LABELS[y_idx])
        ax.set_title(f'Gen {generation:04d} — {_OBJ_LABELS[x_idx]} vs {_OBJ_LABELS[y_idx]}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'gen{generation:04d}_{suffix}.png'), dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Population data persistence
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_population_info(
        population: np.ndarray,
        fitness_matrix: np.ndarray,
        label: str,
        generation: int | None,
        output_dir: str,
) -> None:
    """
    Persist population objective values and chromosomes as CSV + JSON.

    File naming
    -----------
    With generation: ``<label>_gen<generation:04d>.csv`` / ``..._arch.json``
    Without:         ``<label>.csv`` / ``<label>_arch.json``
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = f'{label}_gen{generation:04d}' if generation is not None else label

    pop_size = population.shape[0]
    records, arch_map = [], {}

    for i in range(pop_size):
        arch_str = json.dumps(population[i].tolist(), separators=(',', ':'), cls=_NumpyEncoder)
        arch_hash = hashlib.md5(arch_str.encode()).hexdigest()[:10]
        fitness, err, params, flops = fitness_matrix[i]
        records.append({
            'id': i,
            'Error': round(float(err), 6),
            'Params(M)': round(params / 1e6, 3),
            'FLOPs(M)': round(flops / 1e6, 3),
            'Fitness': round(float(fitness), 6),
            'ArchHash': arch_hash,
        })
        arch_map[arch_hash] = population[i].tolist()

    pd.DataFrame(records).to_csv(os.path.join(output_dir, f'{stem}.csv'), index=False)
    with open(os.path.join(output_dir, f'{stem}_arch.json'), 'w', encoding='utf-8') as f:
        json.dump(arch_map, f, indent=2)


# ---------------------------------------------------------------------------
# ResultSaver
# ---------------------------------------------------------------------------

class ResultSaver:
    """
    Save the best-found architecture as a fully reproducible package.

    Output directory: ``scripts/<n>/``

    Files common to both modes
    --------------------------
    ``__init__.py``      — Entry point loaded by ``train.py`` via ``importlib``;
                           ``Net`` signature is identical in both modes.
    ``chromosome.json``  — Raw chromosome (for restoring search or re-generating code).
    ``search_info.json`` — Search metadata (includes ``net_mode`` field).
    ``train.sh``         — Ready-to-run training script.
    ``README.txt``       — Human-readable summary.

    Static mode (arch implements ``generate_code``)
    ------------------------------------------------
    ``net.py``       — Frozen network definition; ``__init__.py`` re-exports ``Net``.
    ``__init__.py``  — ``from scripts.<n>.net import Net``

    Dynamic mode (arch does not implement ``generate_code``)
    ---------------------------------------------------------
    ``__init__.py``  — Wrapper ``Net`` that subclasses ``arch.Net`` and injects
                       the frozen ``individual`` / ``dataset`` at init time.
                       External signature: ``Net(dropout, drop_connect_rate)``.
                       Advantage: always uses the latest decode logic as the arch evolves.

    Both modes are fully transparent to ``train.py``'s ``_load_net()``.
    """

    SCRIPTS_DIR = 'scripts'

    def save(
            self,
            name: str,
            individual: np.ndarray,
            dataset: str,
            arch_name: str,
            obj_vals: np.ndarray | list,
            search_meta: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a complete search-result package.

        Parameters
        ----------
        name:
            Output subdirectory name, e.g. ``'Best_architecture'``.
        individual:
            Shape ``(num_blocks, num_params)``; a single chromosome.
        dataset:
            Dataset name.
        arch_name:
            Arch module name.
        obj_vals:
            ``[fitness, err, params, flops]``.
        search_meta:
            Additional metadata written to ``search_info.json``.

        Returns
        -------
        str
            Output directory path.
        """
        from archs import load_arch, has_codegen

        out_dir = os.path.join(self.SCRIPTS_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        fitness, err, params, flops = [float(v) for v in obj_vals]

        arch = load_arch(arch_name)
        static = has_codegen(arch)
        mode = 'static' if static else 'dynamic'

        # 1. chromosome.json (both modes)
        self._save_chromosome(individual, out_dir)

        # 2. net.py + __init__.py
        if static:
            self._save_static(arch, individual, dataset, name, out_dir)
        else:
            self._save_dynamic(arch_name, dataset, name, out_dir)

        # 3. search_info.json
        self._save_search_info(
            out_dir, name, arch_name, dataset,
            err, params, flops, mode,
            search_meta or {},
        )

        # 4. train.sh
        self._save_train_sh(out_dir, name, dataset)

        # 5. README.txt
        self._save_readme(out_dir, name, arch_name, dataset, err, params, flops, mode)

        return out_dir

    # ------------------------------------------------------------------
    # Static mode
    # ------------------------------------------------------------------

    @staticmethod
    def _save_static(arch, individual, dataset, name, out_dir):
        """Generate a frozen net.py; __init__.py re-exports Net directly."""
        code = arch.generate_code(individual, dataset)
        with open(os.path.join(out_dir, 'net.py'), 'w', encoding='utf-8') as f:
            f.write(code)

        init_code = textwrap.dedent(f"""            # static mode: network structure frozen in net.py
            from scripts.{name}.net import Net

            __all__ = ['Net']
        """)
        with open(os.path.join(out_dir, '__init__.py'), 'w', encoding='utf-8') as f:
            f.write(init_code)

    # ------------------------------------------------------------------
    # Dynamic mode
    # ------------------------------------------------------------------

    @staticmethod
    def _save_dynamic(arch_name, dataset, name, out_dir):
        """
        Generate a wrapper __init__.py.

        The wrapper subclasses ``arch.Net`` and injects the frozen individual
        and dataset at construction time.  The external signature becomes
        ``Net(dropout, drop_connect_rate)``, matching the static mode.
        ``train.py``'s ``_load_net()`` requires no modification.
        """
        init_code = textwrap.dedent(f"""            # dynamic mode: chromosome decoded at runtime from chromosome.json
            # arch: {arch_name}  dataset: {dataset}
            # To freeze as static code, implement generate_code() for
            # archs.{arch_name} and re-run the search.
            import json
            import numpy as np
            from pathlib import Path
            from archs import load_arch

            _DIR        = Path(__file__).parent
            _arch       = load_arch('{arch_name}')
            _individual = np.array(
                json.loads((_DIR / 'chromosome.json').read_text()),
                dtype=object,
            )
            _dataset    = '{dataset}'


            class Net(_arch.Net):
                \"\"\"
                Dynamic-decode network (wrapper).

                Subclasses arch.Net and injects the frozen individual / dataset
                at init time.  External signature matches the static mode:
                Net(dropout, drop_connect_rate).
                \"\"\"

                def __init__(
                    self,
                    dropout: float           = 0.2,
                    drop_connect_rate: float = 0.0,
                ) -> None:
                    super().__init__(
                        individual        = _individual,
                        dataset           = _dataset,
                        dropout           = dropout,
                        drop_connect_rate = drop_connect_rate,
                    )


            __all__ = ['Net']
        """)
        with open(os.path.join(out_dir, '__init__.py'), 'w', encoding='utf-8') as f:
            f.write(init_code)

    # ------------------------------------------------------------------
    # Shared save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_chromosome(individual: np.ndarray, out_dir: str) -> None:
        with open(os.path.join(out_dir, 'chromosome.json'), 'w', encoding='utf-8') as f:
            json.dump(individual.tolist(), f, indent=2)

    @staticmethod
    def _save_search_info(
            out_dir, name, arch_name, dataset,
            err, params, flops, mode, extra,
    ) -> None:
        info = {
            'name': name,
            'arch': arch_name,
            'dataset': dataset,
            'net_mode': mode,  # 'static' | 'dynamic'
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objectives': {
                'error': round(err, 6),
                'params_M': round(params / 1e6, 3),
                'flops_M': round(flops / 1e6, 3),
            },
            **extra,
        }
        with open(os.path.join(out_dir, 'search_info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _save_train_sh(out_dir: str, name: str, dataset: str) -> None:
        script = textwrap.dedent(f"""            #!/bin/bash
            # Auto-generated by NAS search. Edit hyperparameters as needed.
            python train.py \\
                --script_name        {name} \\
                --dataset            {dataset} \\
                --batch_size         128 \\
                --total_epochs       600 \\
                --warmup_epochs      10 \\
                --lr                 0.05 \\
                --weight_decay       4e-5 \\
                --aug_level          basic \\
                --dropout            0.2 \\
                --drop_connect_rate  0.2
        """)
        path = os.path.join(out_dir, 'train.sh')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(script)
        os.chmod(path, 0o755)

    @staticmethod
    def _save_readme(
            out_dir, name, arch_name, dataset,
            err, params, flops, mode,
    ) -> None:
        mode_note = (
            'Architecture frozen in net.py; no runtime dependency on the archs package.'
            if mode == 'static' else
            'Decoded at runtime from chromosome.json; depends on the archs package '
            '(always uses the latest decode logic as the arch evolves).'
        )
        text = textwrap.dedent(f"""            ============================================================
              NAS Search Result -- {name}
            ============================================================
              Arch      : {arch_name}
              Dataset   : {dataset}
              Net mode  : {mode}  ({mode_note})
              Saved at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

              Objectives
              ----------
              Error     : {err:.4f}
              Params    : {params / 1e6:.2f} M
              FLOPs     : {flops / 1e6:.2f} M

            Files
            -----
              __init__.py      -- train.py load entry (identical interface in both modes)
              chromosome.json  -- Raw chromosome (restore search or regenerate code)
              search_info.json -- Search metadata (includes net_mode)
              train.sh         -- Ready-to-run training script
            {'  net.py            -- Frozen network definition (static mode)' if mode == 'static' else ''}
            Quick Start
            -----------
              bash scripts/{name}/train.sh
        """)
        with open(os.path.join(out_dir, 'README.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
