import numpy as np


def non_dominated_sort(pop_obj: np.ndarray, max_remain: int = None):
    """
    Fast non-dominated sorting (NSGA-II style).

    Parameters
    ----------
    pop_obj:
        Objective matrix of shape ``(n, m)``, where *n* is the population
        size and *m* is the number of objectives.
    max_remain:
        Optional early-exit threshold; sorting stops once at least
        *max_remain* individuals have been assigned a front.

    Returns
    -------
    front_no : np.ndarray of shape ``(n,)``
        Front number for each individual (1 = first / best front).
    max_f_no : int
        Total number of fronts found.
    """
    n, m = pop_obj.shape

    # Initialise; individuals not yet assigned a front carry inf.
    front_no = np.full(n, np.inf, dtype=np.float32)

    # Build (n, n, m) comparison tensors.
    pop_obj_expanded = pop_obj[:, np.newaxis, :]  # (n, 1, m)
    # less[i, j]    = True if individual i is strictly better than j on >= 1 objective
    # greater[i, j] = True if individual i is strictly worse  than j on >= 1 objective
    less    = np.any(pop_obj_expanded < pop_obj[np.newaxis, :, :], axis=2)  # (n, n)
    greater = np.any(pop_obj_expanded > pop_obj[np.newaxis, :, :], axis=2)  # (n, n)

    # dominates_matrix[i, j] = True iff i dominates j:
    #   i <= j on all objectives AND i < j on at least one objective.
    dominates_matrix = less & ~greater

    dominated_count = np.sum(dominates_matrix, axis=0)          # times each j is dominated
    dominates_set   = [np.where(dominates_matrix[i])[0].tolist() for i in range(n)]

    max_f_no      = 0
    current_front = np.where(dominated_count == 0)[0].tolist()  # first (Pareto-optimal) front

    while len(current_front) > 0:
        max_f_no += 1
        for idx in current_front:
            front_no[idx] = max_f_no
        next_front = []
        for idx in current_front:
            for dominated_idx in dominates_set[idx]:
                dominated_count[dominated_idx] -= 1
                if dominated_count[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current_front = next_front
        if max_remain is not None and np.sum(front_no < np.inf) >= max_remain:
            break

    return front_no, max_f_no


def crowding_distance(fitness: np.ndarray) -> np.ndarray:
    """
    Compute the crowding distance for each individual in a front.

    Parameters
    ----------
    fitness:
        Objective matrix of shape ``(n, m)``.

    Returns
    -------
    np.ndarray of shape ``(n,)``
        Crowding distance per individual; boundary individuals receive ``inf``.
    """
    n, m = fitness.shape
    distances = np.zeros(n, dtype=float)
    if n <= 2:
        distances[:] = np.inf
        return distances

    for m in range(m):
        idx      = np.argsort(fitness[:, m])
        f_sorted = fitness[idx, m]
        f_min, f_max = f_sorted[0], f_sorted[-1]
        distances[idx[0]] = distances[idx[-1]] = np.inf

        if f_max == f_min:
            continue  # Avoid division by zero when all values are identical.
        # Vectorised "right_neighbour - left_neighbour" computation.
        distances[idx[1:-1]] += (f_sorted[2:] - f_sorted[:-2]) / (f_max - f_min)

    return distances
