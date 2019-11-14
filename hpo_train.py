"""Hyperparameter optimization of train.py for cifar10_cnn"""
from crayai import hpo

# Set up evaluator
evaluator = hpo.Evaluator('python ./train.py configs/hpo_cifar10_cnn.yaml --hpo',
                          nodes=4,
                          verbose=True)

# Set up search space for HPs: learning rate and dropout
params = hpo.Params([['--optimizer lr', 0.001, (1e-6, 1)],
                     ['--dropout', 0.1, (0.0, 0.5)]])


# Each eval takes ~6 minutes, so 20 evals over 4 nodes should take ~30 minutes
optimizer = hpo.RandomOptimizer(evaluator, num_iters=20, seed=42, verbose=True)

# Optimize the hyperparameters
optimizer.optimize(params)

# Print the figure of merit value for the best set of hyperparameters
print(optimizer.best_fom)
# Print the best set of hyperparameters found
print(optimizer.best_params)

