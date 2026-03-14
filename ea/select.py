import numpy as np


def non_dominated_sort(pop_obj: np.ndarray, max_remain: int = None):
    """
    pop_obj: (n, m) 个体目标矩阵
        n: 种群个体数量
        m: 目标数量
    max_remain: 保留的最大个体数量，可选
    返回：
        front_no: 每个个体的非支配层编号 (0表示第一层)
        max_f_no: 最大层数
    """
    # pop_obj = pop_obj.copy()
    n, m = pop_obj.shape

    # 初始化每个个体的支配计数，-1 表示未分层
    front_no = np.full(n, np.inf, dtype=np.float32)

    # 构建 NxN 的比较矩阵，用于判断谁支配谁
    # pop_obj_expanded[i, j, m] = pop_obj[i, m] 与 pop_obj[j, m] 对比
    pop_obj_expanded = pop_obj[:, np.newaxis, :]  # (n,1,m)
    # less[i,j] = i 在至少一个目标上 < j
    # (n,n,m) 布尔矩阵，每个 [i,j,k] 表示第 i 个个体的第 k 个目标是否小于第 j 个个体的第 k 个目标。
    less = np.any(pop_obj_expanded < pop_obj[np.newaxis, :, :], axis=2)  # (n,n)
    # greater[i,j] = i 在至少一个目标上 > j
    greater = np.any(pop_obj_expanded > pop_obj[np.newaxis, :, :], axis=2)  # (n,n)

    # i 支配 j 的条件：i ≤ j 所有目标且 i < j 至少一个目标
    dominates_matrix = less & ~greater  # True 表示行 i 支配列 j

    # 被支配计数
    dominated_count = np.sum(dominates_matrix, axis=0)  # 每列被支配次数
    # 支配谁
    dominates_set = [np.where(dominates_matrix[i])[0].tolist() for i in range(n)]

    max_f_no = 0
    current_front = np.where(dominated_count == 0)[0].tolist()  # 第一层非支配个体

    while len(current_front) > 0:
        max_f_no += 1
        for idx in current_front:
            front_no[idx] = max_f_no
        next_front = []
        # 对当前层每个个体，更新它支配的个体的被支配计数
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
    参数:
        F: ndarray, shape = (n, m)，每行一个个体的目标值
    返回:
        distances: ndarray, shape = (n,)
    """
    n, m = fitness.shape
    distances = np.zeros(n, dtype=float)
    if n <= 2:
        distances[:] = np.inf
        return distances

    for m in range(m):
        idx = np.argsort(fitness[:, m])
        f_sorted = fitness[idx, m]
        f_min, f_max = f_sorted[0], f_sorted[-1]
        distances[idx[0]] = distances[idx[-1]] = np.inf

        if f_max == f_min:
            continue  # 避免除零
        # 错位相减，一次性计算所有中间个体的“右邻减左邻”差值。
        distances[idx[1:-1]] += (
                (f_sorted[2:] - f_sorted[:-2]) / (f_max - f_min)
        )

    return distances
