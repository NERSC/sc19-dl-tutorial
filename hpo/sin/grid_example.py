#!/usr/bin/env python3
# encoding: utf-8
"""Grid optimizer example"""
from crayai import hpo

evaluator = hpo.Evaluator('python source/sin.py', verbose=True)

params = hpo.Params([["-a", 1.0, (-1.0, 1.0)],
                     ["-b",-1.0, (-1.0, 1.0)],
                     ["-c", 1.0, (-1.0, 1.0)],
                     ["-d",-1.0, (-1.0, 1.0)],
                     ["-e", 1.0, (-1.0, 1.0)],
                     ["-f",-1.0, (-1.0, 1.0)],
                     ["-g", 1.0, (-1.0, 1.0)]])

# Choose 2 grid points from each HP in addition initial point
optimizer = hpo.GridOptimizer(evaluator,
                              grid_size=2,
                              chunk_size=50,
                              verbose=True)

optimizer.optimize(params)

print(optimizer.best_fom)
print(optimizer.best_params)
