import tensorflow as tf
import numpy as np
import time
import keras

from keras.callbacks import Callback
from keras import backend as K
import ml_comm as mc
import math

from cpe_ml import callbacks

#init and finalize
def init():
    mc.init_mpi()
    
def finalize():
    mc.finalize()
    
#some mpicomm features
def size():
    return mc.get_nranks()
    
def rank():
    return mc.get_rank()

class _DistributedOptimizer(keras.optimizers.Optimizer):
    """
    Leveraging approach used in horovod.keras.DistributedOptimizer.
    """

    def __init__(self, name, **kwargs):
        if name is None:
            name = "Distributed%s" % self.__class__.__base__.__name__
        self._name = name
        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        grads = super(self.__class__, self).get_gradients(loss, params)
        grads_mc = mc.gradients(grads, 0)
        return grads_mc

def DistributedOptimizer(optimizer, name=None):
    """
    An optimizer that wraps another keras.optimizers.Optimizer
    """
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, **optimizer.get_config())