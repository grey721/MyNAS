"""
Evolutionary search main logic (Searcher).

Changelog
---------
- Added *arch_name* parameter; population initialisation and gene mutation are
  now delegated to the arch module, so ``ga.py`` has zero knowledge of the
  concrete parameter structure.
- *proxy* parameter changed from a string to a ``BaseProxy`` instance;
  proxy construction is now the caller's responsibility (see ``search.py``).
- Fixed ``mute()`` in-place mutation of the original population array.
- Fixed ``NameError`` at the end of ``evolve()`` when ``generations=0``
  (variable ``i`` undefined).
- Fixed ``AttributeError`` in ``evolve()`` when ``constraints`` is ``None``
  (``constraints.items()`` called on ``None``).
- Eliminated duplicated logging code in ``_print_pop_info()``.
"""

from __future__ import annotations

import copy

import numpy as np

from archs import load_arch
from ea.evaluate import COL_ERR, COL_FITNESS, COL_FLOPS, COL_PARAMS, Evaluator
from ea.proxy import BaseProxy, NASWOT
from ea.select import crowding_distance, non_dominated_sort
from utils import ResultSaver
from utils.logger import Logger


class Searcher:
    """
    Dual-population evolutionary searcher.

    Maintains two populations:
    - **P1**: constraint-elite pool (feasibility-first selection).
    - **P2**: Pareto-diversity pool (NSGA-II-style crowding distance).

    Parameters
    ----------
    arch_name:
        Subdirectory name under ``archs/``, e.g. ``'example_arch'``.
    dataset:
        Dataset name, e.g. ``'cifar10'``.
    batch_size_search:
        Batch size for NAS evaluation.
    p1_size:
        P1 population size (constraint-elite pool).
    p2_size:
        P2 population size (Pareto-diversity pool).
    generations:
        Number of evolutionary generations.
    crossover_rate:
        Crossover probability per pair.
    mutation_rate:
        Per-gene mutation probability.
    constraints:
        Constraint dictionary with format ``{column_index: (min_val, max_val)}``.
        Values are in raw units (bytes / FLOPs), already converted by
        ``search.py``.  ``None`` or empty dict means unconstrained.
    proxy:
        A ``BaseProxy`` instance to use for scoring.
        Construct and pass explicitly::

            from ea.proxy.naswot  import NasWotProxy
            from ea.proxy.synflow import SynFlowProxy

            Searcher(..., proxy=NasWotProxy(batch_size=128))
            Searcher(..., proxy=SynFlowProxy())

        Defaults to ``NasWotProxy(batch_size=batch_size_search)``.
    """

    def __init__(
            self,
            arch_name: str,
            dataset: str,
            batch_size_search: int,
            p1_size: int = 50,
            p2_size: int = 50,
            generations: int = 100,
            crossover_rate: float = 0.7,
            mutation_rate: float = 0.2,
            constraints: dict | None = None,
            proxy: BaseProxy | None = None,
    ) -> None:
        self.log = Logger(name=f'NAS_{arch_name}_{dataset}')

        self.arch_name = arch_name
        self.dataset = dataset
        self.p1_size = p1_size
        self.p2_size = p2_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.constraints = constraints or {}

        # Default proxy when none is supplied.
        if proxy is None:
            proxy = NASWOT(batch_size=batch_size_search)

        arch = load_arch(arch_name)
        self.search_space = arch.get_search_space(dataset)
        self._init_pop_fn = arch.initialize_population
        self._sample_gene_fn = getattr(arch, 'sample_gene', None)

        self.log.info(f'Arch: {arch_name} | Dataset: {dataset} | Proxy: {proxy!r}')
        self.evaluator = Evaluator(arch_name, dataset, batch_size_search, proxy=proxy)
        self.log.info('Searcher initialised.')

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        return self._init_pop_fn(self.search_space, pop_size)

    @staticmethod
    def _tournament(fitness: np.ndarray) -> int:
        pop_size = len(fitness)
        idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
        return int(idx1 if fitness[idx1] >= fitness[idx2] else idx2)

    def crossover(self, population: np.ndarray, fitness: np.ndarray) -> list:
        pop_size = len(population)
        offspring = []

        for _ in range(pop_size // 2):
            if pop_size <= 1:
                self.log.warning('Population size is 1; skipping crossover.')
                return list(population)
            if pop_size == 2:
                idx1, idx2 = 0, 1
            else:
                idx1 = self._tournament(fitness)
                idx2 = self._tournament(fitness)
                while idx2 == idx1:
                    idx2 = self._tournament(fitness)

            p1 = copy.deepcopy(population[idx1])
            p2 = copy.deepcopy(population[idx2])

            if np.random.random() < self.crossover_rate:
                num_genes = len(p1)
                mask = np.random.random(num_genes) > 0.5
                c1 = np.where(mask[:, None], p1, p2)
                c2 = np.where(mask[:, None], p2, p1)
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1, p2])

        return offspring

    def mute(self, population: list) -> list:
        names = self.search_space['names']
        offspring = []

        for indi in population:
            indi = copy.deepcopy(indi)
            for j, name in enumerate(names):
                if np.random.random() < self.mutation_rate:
                    if self._sample_gene_fn is not None:
                        # Delegate to arch-specific sampler (structure-agnostic).
                        indi[j] = self._sample_gene_fn(self.search_space, name)
                    else:
                        # Fallback: sample each candidate list uniformly.
                        indi[j] = [
                            np.random.choice(self.search_space[name][k])
                            for k in range(len(indi[j]))
                        ]
            offspring.append(indi)

        return offspring

    def reproduce(self, population: np.ndarray, fitness: np.ndarray) -> list:
        offspring = self.crossover(population, fitness)
        offspring = self.mute(offspring)
        return offspring

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_p1(
            self,
            candidates: np.ndarray,
            cand_fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        P1 selection: feasibility-first, then fitness-descending within each front.

        Infeasible individuals are ranked by constraint-violation penalty and
        used to fill remaining slots when feasible individuals are insufficient.
        """
        # Deduplicate.
        cand_fitness, unique_idx = np.unique(cand_fitness, axis=0, return_index=True)
        candidates = candidates[unique_idx]

        if not self.constraints:
            # Unconstrained: sort by fitness descending.
            idx = np.argsort(-cand_fitness[:, COL_FITNESS])[:self.p1_size]
            return candidates[idx], cand_fitness[idx]

        # Non-dominated sort on objective columns (exclude fitness column).
        obj_vals = cand_fitness[:, 1:]
        front_no, max_front = non_dominated_sort(obj_vals, obj_vals.shape[0])

        selected = []
        for lvl in range(1, max_front + 1):
            front_idx = np.where(front_no == lvl)[0]
            front_fitness = cand_fitness[front_idx]

            # Constraint satisfaction mask.
            mask = np.ones(len(front_idx), dtype=bool)
            for col, (lo, hi) in self.constraints.items():
                if lo is not None:
                    mask &= front_fitness[:, col] >= lo
                if hi is not None:
                    mask &= front_fitness[:, col] <= hi

            selected.extend(front_idx[mask].tolist())

            if len(selected) >= self.p1_size:
                sel = np.array(selected)
                best = np.argsort(-cand_fitness[sel, COL_FITNESS])[:self.p1_size]
                final = sel[best]
                return candidates[final], cand_fitness[final]

        # Not enough feasible individuals: fill with least-violating infeasible ones.
        sel = np.array(selected) if selected else np.array([], dtype=int)
        not_sel_idx = np.setdiff1d(np.arange(len(candidates)), sel)
        not_sel_fit = cand_fitness[not_sel_idx]

        penalty = np.zeros(len(not_sel_idx), dtype=np.float32)
        for col, (lo, hi) in self.constraints.items():
            v = not_sel_fit[:, col]
            if lo is not None:
                penalty += np.square(np.maximum(lo - v, 0))
            if hi is not None:
                penalty += np.square(np.maximum(v - hi, 0))

        n_fill = self.p1_size - len(sel)
        fill_idx = not_sel_idx[np.argsort(penalty)[:n_fill]]

        final = np.concatenate([sel, fill_idx]) if len(sel) > 0 else fill_idx
        return candidates[final], cand_fitness[final]

    def select_p2(
            self,
            candidates: np.ndarray,
            cand_fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Deduplicate.
        cand_fitness, unique_idx = np.unique(cand_fitness, axis=0, return_index=True)
        candidates = candidates[unique_idx]

        obj_vals = cand_fitness[:, 1:]
        front_no, max_front = non_dominated_sort(obj_vals, obj_vals.shape[0])

        selected = []
        for lvl in range(1, max_front + 1):
            front_idx = np.where(front_no == lvl)[0]

            if len(selected) + len(front_idx) > self.p2_size:
                remaining = self.p2_size - len(selected)
                crowd = crowding_distance(obj_vals[front_idx])
                top_idx = np.argsort(-crowd)[:remaining]
                selected.extend(front_idx[top_idx].tolist())
                break

            selected.extend(front_idx.tolist())
            if len(selected) >= self.p2_size:
                break

        sel = np.array(selected[:self.p2_size])
        return candidates[sel], cand_fitness[sel]

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _print_pop_info(
            self,
            fitness: np.ndarray,
            label: str,
            gen_info: str,
    ) -> None:
        self.log.info(f'{gen_info}{label} size: {len(fitness)}')
        best_idx = np.argmax(fitness[:, COL_FITNESS])
        best = fitness[best_idx, 1:]
        avg = np.mean(fitness[:, 1:], axis=0)
        self.log.info(
            f'{gen_info}{label} best  -> '
            f'Err: {best[0]:.4f}, '
            f'Params: {best[1] / 1e6:.2f} M, '
            f'FLOPs: {best[2] / 1e6:.2f} M'
        )
        self.log.info(
            f'{gen_info}{label} mean  -> '
            f'Err: {avg[0]:.4f}, '
            f'Params: {avg[1] / 1e6:.2f} M, '
            f'FLOPs: {avg[2] / 1e6:.2f} M'
        )

    def _print_p1_p2_info(
            self,
            p1_fitness: np.ndarray,
            p2_fitness: np.ndarray,
            gen_info: str,
    ) -> None:
        self._print_pop_info(p1_fitness, 'P1', gen_info)
        self._print_pop_info(p2_fitness, 'P2', gen_info)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def evolve(self, file_name: str = 'Best_architecture') -> None:
        self.log.info(f'[{file_name}] Starting evolution | Arch: {self.arch_name}')

        # Initialise populations.
        p1 = self._initialize_population(self.p1_size)
        p2 = self._initialize_population(self.p2_size)
        self.log.info('Population initialised.')

        p1_fitness = self.evaluate(p1)
        p2_fitness = self.evaluate(p2)
        self.log.info('Initial fitness evaluation complete.')

        w = len(str(self.generations))
        gen_info = f'GEN [ {0:>{w}} / {self.generations} ] | '
        self._print_p1_p2_info(p1_fitness, p2_fitness, gen_info)
        self.log.plot_pop(p1_fitness, p2_fitness, generation=0)

        last_gen = 0  # Guard against generations=0 (variable i undefined).
        for i in range(1, self.generations + 1):
            last_gen = i
            gen_info = f'GEN [ {i:>{w}} / {self.generations} ] | '

            # Generate offspring.
            p1_off = self.reproduce(p1, p1_fitness[:, COL_FITNESS])
            p2_off = self.reproduce(p2, p2_fitness[:, COL_FITNESS])
            self.log.info(
                f'{gen_info}Offspring generated | '
                f'P1 offspring: {len(p1_off)}, P2 offspring: {len(p2_off)}'
            )

            # Evaluate offspring.
            p1_off_fit = self.evaluate(p1_off)
            p2_off_fit = self.evaluate(p2_off)
            self.log.info(f'{gen_info}Offspring evaluation complete.')

            # Convert offspring lists to arrays for concatenation.
            p1_off_arr = np.array(p1_off)
            p2_off_arr = np.array(p2_off)

            # Select next generation.
            p1_cand = np.concatenate([p1, p1_off_arr, p2_off_arr], axis=0)
            p1_cand_fit = np.concatenate([p1_fitness, p1_off_fit, p2_off_fit], axis=0)
            p1, p1_fitness = self.select_p1(p1_cand, p1_cand_fit)

            p2_cand = np.concatenate([p2, p1_off_arr, p2_off_arr], axis=0)
            p2_cand_fit = np.concatenate([p2_fitness, p1_off_fit, p2_off_fit], axis=0)
            p2, p2_fitness = self.select_p2(p2_cand, p2_cand_fit)

            self.log.info(f'{gen_info}Selection complete.')
            self._print_p1_p2_info(p1_fitness, p2_fitness, gen_info)
            self.log.plot_pop(p1_fitness, p2_fitness, generation=i)

        self.log.info('Evolution finished.')
        self.log.save_population(p1, p1_fitness, label='p1', generation=last_gen)
        self.log.save_population(p2, p2_fitness, label='p2', generation=last_gen)

        # Save the best architecture together with search metadata.
        search_meta = {
            'proxy': repr(self.evaluator.proxy),
            'generations': self.generations,
            'p1_size': self.p1_size,
            'p2_size': self.p2_size,
            'exp_dir': self.log.exp_dir,
        }
        self._save_best(p1, p1_fitness, file_name, search_meta=search_meta)

    def _save_best(
            self,
            p1: np.ndarray,
            p1_fitness: np.ndarray,
            file_name: str,
            search_meta: dict | None = None,
    ) -> None:
        sorted_idx = np.argsort(p1_fitness[:, COL_ERR])

        for idx in sorted_idx:
            vals = p1_fitness[idx]
            valid = True

            if self.constraints:
                for col, (lo, hi) in self.constraints.items():
                    if lo is not None and vals[col] < lo:
                        valid = False
                        break
                    if hi is not None and vals[col] > hi:
                        valid = False
                        break

            if valid:
                self.log.info(
                    f'Selected individual idx={idx}: '
                    f'err={vals[COL_ERR]:.4f}, '
                    f'params={vals[COL_PARAMS] / 1e6:.2f} M, '
                    f'flops={vals[COL_FLOPS] / 1e6:.2f} M'
                )
                saver = ResultSaver()
                out_dir = saver.save(
                    name=file_name,
                    individual=p1[idx],
                    dataset=self.dataset,
                    arch_name=self.arch_name,
                    obj_vals=vals,
                    search_meta=search_meta,
                )
                self.log.info(f'Search result package saved -> {out_dir}')
                return

        self.log.info('No individual satisfies all constraints.')

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return self.evaluator.evaluate(population)
