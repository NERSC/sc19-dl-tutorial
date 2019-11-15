"""Hyperparameter optimization of train.py for cifar10_cnn"""
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
    evaluator = hpo.Evaluator('python ./train.py configs/hpo_cifar10_cnn.yaml --hpo',
                              nodes=args.num_nodes,
                              verbose=args.verbose)

    # Set up search space for HPs: learning rate and dropout
    params = hpo.Params([['--optimizer lr', 0.001, (1e-6, 1)],
                         ['--dropout', 0.1, (0.0, 0.5)]])


    # Each eval takes ~6 minutes, so 20 evals over 4 nodes should take ~30 minutes
    optimizer = hpo.RandomOptimizer(evaluator,
                                    num_iters=1,
                                    verbose=args.verbose)

    # Optimize the hyperparameters
    optimizer.optimize(params)

    # Print the figure of merit value for the best set of hyperparameters
    print(optimizer.best_fom)
    # Print the best set of hyperparameters found
    print(optimizer.best_params)


if __name__ == '__main__':
    main()
