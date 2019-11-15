import argparse

from crayai import hpo

from os import path
from os.path import abspath, dirname

# Give abs path to src directory so that this script can be run from anywhere:
pwd = path.dirname(path.abspath(__file__))
src_path = path.join(pwd, 'source')
run_path = path.join(pwd, 'run')

argparser = argparse.ArgumentParser()
argparser.add_argument('-N', '--num_nodes', type=int, default=1)
argparser.add_argument('--generations', type=int, default=2)
argparser.add_argument('--num_demes', type=int, default=1)
argparser.add_argument('--pop_size', type=int, default=4)
argparser.add_argument('--mutation_rate', type=float, default=0.05)
argparser.add_argument('--crossover_rate', type=float, default=0.33)
argparser.add_argument('--verbose', action='store_true')
args = argparser.parse_args()

print("------------------------------------------------------------")
print("Genetic HPO Example: LeNet-5 (MNIST) TensorFlow -- Cray Inc.")
print("------------------------------------------------------------")

evaluator = hpo.Evaluator('python3 {0}/mnist.py'.format(src_path),
                          run_path=run_path,
                          src_path=src_path,
                          nodes=args.num_nodes)

optimizer = hpo.GeneticOptimizer(evaluator,
                                  generations=args.generations,
                                  num_demes=args.num_demes,
                                  pop_size=args.pop_size,
                                  mutation_rate=args.mutation_rate,
                                  crossover_rate=args.crossover_rate,
                                  verbose=args.verbose,
                                  log_fn='mnist-topology.log')

params = hpo.Params([["--dropout",    0.5,     (0.005,  0.9)],
                     ["--momentum",   1.0e-4,  (1.0e-6, 1.0e-2)],
                     ["--c1_sz",      5,       (2, 8)],
                     ["--c1_ft",      32,      (8, 128)],
                     ["--c2_sz",      5,       (2, 8)],
                     ["--c2_ft",      64,      (16, 256)],
                     ["--fullyc_sz",  1024,    (64, 4096)]])

optimizer.optimize(params)

# Print best FoM value
print('Best FoM: ', optimizer.best_fom)

# Print best hyperparameters
print('Best HPs:')
for param, value in optimizer.best_params.items():
    print(param, ' = ', value)

print("------------------------------------------------------------")
print("Done.")
print("------------------------------------------------------------")
