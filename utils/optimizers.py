"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math

# Externals
import keras

def get_optimizer(name, lr, lr_scaling='linear', n_ranks=1, dist_wrapper=None, **opt_args):
    """
    Configure the optimizer and scale the learning rate by n_ranks.
    """
    # Scale the learning rate
    if lr_scaling == 'linear':
        lr = lr * n_ranks
    elif lr_scaling == 'sqrt':
        lr = lr * math.sqrt(n_ranks)

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    # Distributed optimizer wrapper
    if n_ranks > 1 and dist_wrapper:
        opt = dist_wrapper(opt)

    return opt
