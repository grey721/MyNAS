import copy

import numpy as np

from ea.evaluate import Evaluator
from ea.genotypes import search_space_cifar10
from ea.select import non_dominated_sort, crowding_distance
from utils.logger import Logger
from utils import SaveFile


class Searcher:
    def __init__(self, dataset, batch_size_search,
                 p1_size=50, p2_size=50, generations=100,
                 crossover_rate=0.7, mutation_rate=0.2,
                 constraint=None,
                 random_seed=42,
                 ):
        self.log = Logger()
        self.p1_size = p1_size
        self.p2_size = p2_size
        self.generations = generations

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.search_space = self._get_search_space(dataset)
        self.log.info(f'Dataset: {dataset}')
        self.dataset = dataset
        self.constraint = constraint

        self.log.info(f'Random seed: {random_seed}')
        self.evaluator = Evaluator(dataset, batch_size_search, random_seed=random_seed)
        self.log.info('Searcher initialized.')

    @staticmethod
    def _get_search_space(dataset):
        if 'cifar' in dataset:
            return search_space_cifar10
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def _initialize_population(self, pop_size):
        names = self.search_space['names']
        num_blocks = len(names)
        num_params = 6  # [s, e, b, k, se, f]
        population = np.empty((pop_size, num_blocks, num_params), dtype=object)

        rand_len = np.random.rand(pop_size)
        rand_s = np.random.rand(pop_size, num_blocks)

        for j, name in enumerate(names):
            s_candidates = self.search_space[name][0]
            e_candidates = self.search_space[name][1]
            b_candidates = self.search_space[name][2]
            k_candidates = self.search_space[name][3]
            se_candidates = self.search_space[name][4]
            f_candidates = self.search_space[name][5]

            # 处理 s
            if 0 in s_candidates:
                s_nonzero = [x for x in s_candidates if x != 0]
                s_idx = (rand_s[:, j] > rand_len)
                s_values = np.zeros(pop_size, dtype=object)
                choose_mask = ~s_idx
                s_values[choose_mask] = np.random.choice(s_nonzero, size=np.sum(choose_mask))
            else:
                s_values = np.random.choice(s_candidates, size=pop_size)

            # 其他参数采样
            e_values = np.random.choice(e_candidates, size=pop_size)
            b_values = np.random.choice(b_candidates, size=pop_size)
            k_values = np.random.choice(k_candidates, size=pop_size)
            se_values = np.random.choice(se_candidates, size=pop_size)
            f_values = np.random.choice(f_candidates, size=pop_size)

            # 汇总
            population[:, j, 0] = s_values
            population[:, j, 1] = e_values
            population[:, j, 2] = b_values
            population[:, j, 3] = k_values
            population[:, j, 4] = se_values
            population[:, j, 5] = f_values

        return population

    @staticmethod
    def _choose_one_parent(fitness):
        pop_size = len(fitness)
        idx1 = np.random.randint(0, pop_size)
        idx2 = np.random.randint(0, pop_size)
        while idx2 == idx1:
            idx2 = np.random.randint(0, pop_size)

        if fitness[idx1] >= fitness[idx2]:
            return idx1
        else:
            return idx2

    def _print_p1_p2_info(self, p1_fitness, p2_fitness, gen_info):
        self.log.info(f'{gen_info}New P1 number: {len(p1_fitness)}')
        best_idx_1 = np.argmax(p1_fitness[:, 0])
        best_1 = p1_fitness[best_idx_1, 1:]  # error, params, flops
        avg_1 = np.mean(p1_fitness[:, 1:], axis=0)
        self.log.info(
            f'{gen_info}P1 Best Individual -> Error: {best_1[0]:.4f}, Params: {best_1[1] / 1e6:.2f} M, FLOPs: {best_1[2] / 1e6:.2f} M')
        self.log.info(
            f'{gen_info}P1 Avg -> Error: {avg_1[0]:.4f}, Params: {avg_1[1] / 1e6:.2f} M, FLOPs: {avg_1[2] / 1e6:.2f} M')

        self.log.info(f'{gen_info}New P2 number: {len(p2_fitness)}')
        best_idx_2 = np.argmax(p2_fitness[:, 0])
        best_2 = p2_fitness[best_idx_2, 1:]
        avg_2 = np.mean(p2_fitness[:, 1:], axis=0)
        self.log.info(
            f'{gen_info}P2 Best Individual -> Error: {best_2[0]:.4f}, Params: {best_2[1] / 1e6:.2f} M, FLOPs: {best_2[2] / 1e6:.2f} M')
        self.log.info(
            f'{gen_info}P2 Avg -> Error: {avg_2[0]:.4f}, Params: {avg_2[1] / 1e6:.2f} M, FLOPs: {avg_2[2] / 1e6:.2f} M')

    def evaluate(self, population):
        return self.evaluator.evaluate(population)

    def mute(self, population):
        offspring = []
        for indi in population:
            names = self.search_space['names']
            for mutate_point, name in enumerate(names):
                if np.random.random() < self.mutation_rate:
                    indi[mutate_point] = [
                        float(np.random.choice(self.search_space[name][0])),
                        int(np.random.choice(self.search_space[name][1])),
                        int(np.random.choice(self.search_space[name][2])),
                        int(np.random.choice(self.search_space[name][3])),
                        float(np.random.choice(self.search_space[name][4])),
                        float(np.random.choice(self.search_space[name][5]))
                    ]
            offspring.append(indi)
        return offspring

    def crossover(self, population, fitness):
        pop_size = len(population)
        offspring = []
        for _ in range(pop_size // 2):
            idx1 = self._choose_one_parent(fitness)
            idx2 = self._choose_one_parent(fitness)
            while idx2 == idx1:
                if pop_size == 2:
                    idx1 = 0
                    idx2 = 1
                elif pop_size == 1:
                    self.log.warning('pop_size = 1')
                    return [population[0]]
                else:
                    idx2 = self._choose_one_parent(fitness)
            assert idx1 < pop_size and idx2 < pop_size

            parent1, parent2 = copy.deepcopy(population[idx1]), copy.deepcopy(population[idx2])

            if np.random.random() < self.crossover_rate:
                offspring1 = []
                offspring2 = []
                num_genes = len(parent1)
                mask = np.random.random(num_genes)
                for i in range(num_genes):
                    if mask[i] <= 0.5:
                        offspring1.append(parent1[i])
                        offspring2.append(parent2[i])
                    else:
                        offspring1.append(parent2[i])
                        offspring2.append(parent1[i])

                offspring.append(offspring1)
                offspring.append(offspring2)
            else:
                offspring.append(parent1)
                offspring.append(parent2)

        return offspring

    def reproduce(self, population, fitness):
        offspring = self.crossover(population, fitness)
        offspring = self.mute(offspring)
        return offspring

    def select_p1(self, p1_candidate, p1_candidate_fitness):
        # 去重
        p1_candidate_fitness, unique_indices = np.unique(p1_candidate_fitness, axis=0, return_index=True)
        p1_candidate = p1_candidate[unique_indices]

        # 检查约束是否有效
        has_constraint = any(not (obj_min == 0 and obj_max == 0) for obj_min, obj_max in self.constraint.values())
        if not has_constraint:
            # 无约束，直接按适应度排序
            sorted_idx = np.argsort(-p1_candidate_fitness[:, 0])[:self.p1_size]
            return p1_candidate[sorted_idx], p1_candidate_fitness[sorted_idx]

        # 非支配排序（排除适应度列）
        p1_obj_value = p1_candidate_fitness[:, 1:]
        front_temp, max_front = non_dominated_sort(p1_obj_value, p1_obj_value.shape[0])

        selected_indices = []

        # 遍历前沿层，按层挑选满足约束的个体
        for front_level in range(1, max_front + 1):
            front_indices = np.where(front_temp == front_level)[0]
            front_fitness = p1_candidate_fitness[front_indices]

            # 构建约束掩码
            mask = np.ones(len(front_indices), dtype=bool)
            for obj_idx, (obj_min, obj_max) in self.constraint.items():
                if obj_min == 0 and obj_max == 0:
                    continue
                mask &= (front_fitness[:, obj_idx] >= obj_min) & (front_fitness[:, obj_idx] <= obj_max)

            constrained_indices = front_indices[mask]
            selected_indices.extend(constrained_indices)

            # 如果数量够了，直接按适应度排序裁剪
            if len(selected_indices) >= self.p1_size:
                selected_indices = np.array(selected_indices)
                best_indices = np.argsort(-p1_candidate_fitness[selected_indices, 0])[:self.p1_size]
                final_idx = selected_indices[best_indices]
                return p1_candidate[final_idx], p1_candidate_fitness[final_idx]

        # 数量不足，从未选择个体中挑选违约最小的
        selected_indices = np.array(selected_indices)
        not_selected_idx = np.setdiff1d(np.arange(len(p1_candidate)), selected_indices)
        not_selected_fitness = p1_candidate_fitness[not_selected_idx]

        # 计算约束违约平方和
        penalty = np.zeros(len(not_selected_idx), dtype=np.float32)
        for obj_idx, (obj_min, obj_max) in self.constraint.items():
            if obj_min == 0 and obj_max == 0:
                continue
            val = not_selected_fitness[:, obj_idx]
            penalty += np.square(np.maximum(obj_min - val, 0))
            penalty += np.square(np.maximum(val - obj_max, 0))

        remaining_num = self.p1_size - len(selected_indices)
        fill_idx_order = np.argsort(penalty)[:remaining_num]
        fill_idx = not_selected_idx[fill_idx_order]

        # 拼接已选择的个体与填充个体
        if len(selected_indices) > 0:
            final_idx = np.hstack([selected_indices, fill_idx])
        else:
            final_idx = fill_idx

        return p1_candidate[final_idx], p1_candidate_fitness[final_idx]

    def select_p2(self, p2_candidate, p2_candidate_fitness):
        # 去重
        p2_candidate_fitness, unique_indices = np.unique(p2_candidate_fitness, axis=0, return_index=True)
        p2_candidate = p2_candidate[unique_indices]

        # 非支配排序（排除适应度列）
        p2_obj_value = p2_candidate_fitness[:, 1:]
        front_temp, max_front = non_dominated_sort(p2_obj_value, p2_obj_value.shape[0])

        selected_indices = []

        # 遍历每一前沿层
        for front_level in range(1, max_front + 1):
            front_indices = np.where(front_temp == front_level)[0]

            # 如果加上当前层就超过上限，则在该层内部用拥挤距离筛选
            if len(selected_indices) + len(front_indices) > self.p2_size:
                remaining_num = self.p2_size - len(selected_indices)
                front_objs = p2_obj_value[front_indices]
                crowd_dist = crowding_distance(front_objs)
                sorted_idx = np.argsort(-crowd_dist)[:remaining_num]  # 拥挤度大的优先
                selected_indices.extend(front_indices[sorted_idx])
                break
            else:
                selected_indices.extend(front_indices)

            if len(selected_indices) >= self.p2_size:
                break

        selected_indices = np.array(selected_indices[:self.p2_size])
        return p2_candidate[selected_indices], p2_candidate_fitness[selected_indices]

    def evolve(self, file_name='Best_architecture'):
        # 初始化种群
        self.log.info(f'[{file_name}] Begin evolution.')
        p1 = self._initialize_population(self.p1_size)
        p2 = self._initialize_population(self.p2_size)
        self.log.info('Population initialized.')

        # 评估初始种群
        p1_fitness = self.evaluate(p1)
        p2_fitness = self.evaluate(p2)
        self.log.info('Fitness initialized.')

        # 初始信息
        self._print_p1_p2_info(p1_fitness, p2_fitness, gen_info=f'GEN{0:>4} | ')
        self.log.plot_pop(p1_fitness, p2_fitness, generation=0)

        for i in range(1, self.generations + 1):
            gen_info = f'GEN{i:>4} | '

            # 生成子代
            self.log.info(f'{gen_info}Generating offspring.')
            p1_offspring = self.reproduce(p1, p1_fitness[:, 0])
            p2_offspring = self.reproduce(p2, p2_fitness[:, 0])
            self.log.info(f'{gen_info}Generation completed.')
            self.log.info(f'{gen_info}P1 Offspring number: {len(p1_offspring)}')
            self.log.info(f'{gen_info}P2 Offspring number: {len(p2_offspring)}')

            # 评估后代
            self.log.info(f'{gen_info}Begin fitness evaluation.')
            p1_offspring_fitness = self.evaluate(p1_offspring)
            p2_offspring_fitness = self.evaluate(p2_offspring)
            self.log.info(f'{gen_info}Evaluation completed.')

            # 选择下一代
            self.log.info(f'{gen_info}Begin population selection.')
            p1_candidate = np.concatenate([p1, p1_offspring, p2_offspring], axis=0)
            p1_candidate_fitness = np.concatenate([p1_fitness, p1_offspring_fitness, p2_offspring_fitness], axis=0)
            p1, p1_fitness = self.select_p1(p1_candidate, p1_candidate_fitness)

            p2_candidate = np.concatenate([p2, p1_offspring, p2_offspring], axis=0)
            p2_candidate_fitness = np.concatenate([p2_fitness, p1_offspring_fitness, p2_offspring_fitness], axis=0)
            p2, p2_fitness = self.select_p2(p2_candidate, p2_candidate_fitness)
            self.log.info(f'{gen_info}Selection completed.')
            self._print_p1_p2_info(p1_fitness, p2_fitness, gen_info)

            # 可视化
            self.log.plot_pop(p1_fitness, p2_fitness, generation=i)

        self.log.info('Evolution completed.')
        self.log.save_population(p1, p1_fitness, filename='p1', generation=i)
        self.log.save_population(p2, p2_fitness, filename='p2', generation=i)

        # 按 err
        sorted_idx = np.argsort(p1_fitness[:, 1])
        for idx in sorted_idx:
            obj_vals = p1_fitness[idx]
            valid = True
            for obj_idx, (obj_min, obj_max) in self.constraint.items():
                if obj_min == 0 and obj_max == 0:
                    continue  # 未设置约束则跳过
                # print(f'{obj_min}<={obj_vals[obj_idx]}<={obj_max}？')
                if not (obj_min <= obj_vals[obj_idx] <= obj_max):
                    valid = False
                    break

            if valid:
                self.log.info(
                    f"Selected individual idx={idx}: "
                    f"err={obj_vals[1]:.4f}, params={obj_vals[2] / 1e6:.2f}M, adds={obj_vals[3] / 1e6:.2f}M"
                )

                sf = SaveFile()
                sf.set_dataset(self.dataset)
                sf.set_architecture_with_individual(p1[idx])
                sf.generate_py(file_name=file_name, project_name=self.log.project_name)

                self.log.info(f"Model Script saved to directory: scripts/{file_name}")
                break
        else:
            self.log.info("No individual satisfies the constraint conditions.")


