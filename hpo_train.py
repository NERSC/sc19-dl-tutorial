"""Hyperparameter optimization of cifar10_cnn example (train.py)"""

# System
import argparse

# Externals
from crayai import hpo

def parse_args():
    parser = argparse.ArgumentParser('hpo_train.py')
    parser.add_argument('-N', '--num_nodes', type=int, default=1,
                        help='number of nodes to evaluate over')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose HPO output')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up evaluator
    # configs/hpo_cifar10_cnn.yaml is a scaled down version of cifar10
    # --hpo is required for train.py to print the FoM
    # --no-output is required to avoid checkpointing
    eval_cmd = 'python ./train.py configs/hpo_cifar10_cnn.yaml --hpo --no-output'
    evaluator = hpo.Evaluator(eval_cmd,
                              nodes=args.num_nodes,
                              verbose=args.verbose)

    # Set up search space for HPs: learning rate and dropout
    params = hpo.Params([['--optimizer lr', 0.001, (1e-6, 1)],
                         ['--dropout', 0.1, (0.0, 0.5)]])


    # Set up genetic optimizer with 16 evaluations/generation and 3 generations
    optimizer = hpo.GeneticOptimizer(evaluator, generations=3, pop_size=16,
                                     num_demes=1, mutation_rate=0.6,
                                     verbose=args.verbose)

    # Optimize the hyperparameters
    optimizer.optimize(params)

    # Print the figure of merit value for the best set of hyperparameters
    print(optimizer.best_fom)
    # Print the best set of hyperparameters found
    print(optimizer.best_params)


if __name__ == '__main__':
    main()
