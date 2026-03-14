import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ea.ga import Searcher


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary Architecture Search")

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--p1_size', type=int, default=3, help='Population 1 size')
    parser.add_argument('--p2_size', type=int, default=3, help='Population 2 size')
    parser.add_argument('--num_generations', type=int, default=2, help='Number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.2)
    parser.add_argument('--crossover_rate', type=float, default=0.8)
    parser.add_argument('--batch_size_search', type=int, default=128)

    parser.add_argument('--params_max', type=float, default=None, help='Maximum parameters in M (e.g., 0.5)')
    parser.add_argument('--params_min', type=float, default=None, help='Minimum parameters in M')
    parser.add_argument('--flops_max', type=float, default=None, help='Maximum FLOPs in M (e.g., 200)')
    parser.add_argument('--flops_min', type=float, default=None, help='Minimum FLOPs in M')

    return parser.parse_args()


def check_directory():
    dirs = ['./logs', './scripts', './trained_models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    check_directory()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    constraints = {2: (args.params_min, args.params_max),
                   3: (args.flops_min, args.flops_max)}

    ea = Searcher(
        dataset=args.dataset,
        batch_size_search=args.batch_size_search,
        p1_size=args.p1_size,
        p2_size=args.p2_size,
        generations=args.num_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        constraints=constraints,
    )
    ea.evolve()
