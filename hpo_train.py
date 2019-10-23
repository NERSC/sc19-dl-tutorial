"""Hyperparameter optimization of train.py for cifar10_cnn"""
from crayai import hpo

import numpy as np

#import argparse
#
#parser = argparse.ArgumentParser(description=__doc__,
#                                 formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument('--source', type=str, default='sin.py',
#                    help='source script')
#
#args = parser.parse_args()

evaluator = hpo.Evaluator('python {0}/train.py'.format(args.source), 
                          verbose=True)

"""
HPs:
--dropout: 0.1
# TODO: expose --lr via argparse
--lr: 0.001
--batch_size: 64
"""
params = hpo.Params([['--lr', 0.001, tuple(np.logspace(-6, 0, num=1000))],
                     ['--dropout', 0.1, [0.0, 0.5]])


optimizer = hpo.RandomOptimizer(evaluator, verbose=True, num_iters=10, seed=42)

optimizer.optimize(params)

print(optimizer.best_fom)
print(optimizer.best_params)

