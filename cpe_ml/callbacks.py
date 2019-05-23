import tensorflow as tf
import numpy as np
import time
import keras

from keras.callbacks import Callback
from keras import backend as K
import ml_comm as mc
import math


class InitPluginCallback(Callback):
    def __init__(self, max_steps, buffer_size):
        import ml_comm as mc
        super(InitPluginCallback, self).__init__()
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        
    def on_train_begin(self, logs=None):
        mc.init(1, 1, self.buffer_size, "tensorflow")
        #this enables pipelining after certain number of steps. it screws model accuracy here though
        #mc.config_team(0, 0, int(0.2*self.max_steps), self.max_steps, 0, 100)
        mc.config_team(0, 0, self.max_steps, self.max_steps, 0, 100)


class BroadcastGlobalVariablesCallback(Callback):
    def __init__(self, head_rank, validate=False):
        import ml_comm as mc
        super(BroadcastGlobalVariablesCallback, self).__init__()
        self.head_rank = head_rank
        self.validate  = validate
        
    def on_train_begin(self, logs=None):
        sess = K.get_session()
        
        # Split variables based on type -> float32 vs all else
        test_v = tf.Variable([0], dtype=tf.float32)
        all_vars = tf.trainable_variables()
        float_vars = [v for v in all_vars if v.dtype == test_v.dtype]
        other_vars = [v for v in all_vars if v.dtype != test_v.dtype]

        # Initialize variables and broadcast from head node
        sess.run(tf.variables_initializer(all_vars))
        new_vars = mc.broadcast(float_vars, 0)
        bcast = tf.group(*[tf.assign(v, new_vars[k]) for k,v in enumerate(float_vars)])
        sess.run(bcast)

        # Validate Broadcast
        if self.validate:
            py_all_vars = [sess.run(v) for v in float_vars]
            var_types = [np.array([v]) if type(v) == np.float32 else v for v in py_all_vars]
            if mc.get_rank() is 0:
                if (mc.check_buffers_match(var_types, 1) != 0):
                    tf.logging.error("Not all processes have the same initial model!")
                else:
                    tf.logging.info("Initial model is consistent on all ranks")


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


class MetricAverageCallback(Callback):
    def __init__(self, device='', *args):
        super(MetricAverageCallback, self).__init__(*args)
        self.backend = K
        self.device = device

    def _average_metrics_in_place(self, logs):
        logs = logs or {}
        # Reduce every metric among workers. Sort metrics by name
        # to ensure consistent order.
        #create list of metrics:
        if logs:
            metric_array = np.zeros(len(list(logs.items())), dtype=np.float32)

            #extract metrics and pack into buffer
            for idx, token in enumerate(sorted(logs.items())):
                metric, value = token
                metric_array[idx] = np.float32(value)

            #average array
            mc.average(metric_array)

            # Unpack buffer
            for idx, token in enumerate(sorted(logs.items())):
                metric, _ = token
                logs[metric] = metric_array[idx]

    def on_epoch_end(self, epoch, logs=None):
        self._average_metrics_in_place(logs)


class LearningRateScheduleCallback(Callback):
    def __init__(self, multiplier, start_epoch=0, end_epoch=None, staircase=True,
                 momentum_correction=True, steps_per_epoch=None, *args):
        super(LearningRateScheduleCallback, self).__init__(*args)
        self.backend = K
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.staircase = staircase
        self.momentum_correction = momentum_correction
        self.initial_lr = None
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = None

        if not callable(multiplier):
            self.staircase = True
            self.multiplier = lambda epoch: multiplier
        else:
            self.multiplier = multiplier

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter to the '
                             '%s() or upgrade to the latest version of Keras.'
                             % self.__class__.__name__)

    def _adjust_learning_rate(self, epoch):
        old_lr = self.backend.get_value(self.model.optimizer.lr)
        new_lr = self.initial_lr * self.multiplier(epoch)
        self.backend.set_value(self.model.optimizer.lr, new_lr)

        if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
            # See the paper cited above for more information about momentum correction.
            self.restore_momentum = self.backend.get_value(self.model.optimizer.momentum)
            self.backend.set_value(self.model.optimizer.momentum, self.restore_momentum * new_lr / old_lr)

    def _restore_momentum_if_needed(self):
        if self.restore_momentum:
            self.backend.set_value(self.model.optimizer.momentum, self.restore_momentum)
            self.restore_momentum = None

    def on_train_begin(self, logs=None):
        self.initial_lr = self.backend.get_value(self.model.optimizer.lr)
        if not self.staircase and not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if (self.current_epoch < self.start_epoch or
                (self.end_epoch is not None and self.current_epoch >= self.end_epoch)):
            # Outside of the adjustment scope.
            return

        if self.staircase and batch == 0:
            # Do on first batch of every epoch.
            self._adjust_learning_rate(self.current_epoch)
        elif not self.staircase:
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            self._adjust_learning_rate(epoch)

    def on_batch_end(self, batch, logs=None):
        self._restore_momentum_if_needed()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log current learning rate.
            logs['lr'] = self.backend.get_value(self.model.optimizer.lr)
            

class LearningRateWarmupCallback(LearningRateScheduleCallback):
    def __init__(self, warmup_epochs=5, momentum_correction=True, steps_per_epoch=None,
                 verbose=0, *args):
        def multiplier(epoch):
            # Adjust epoch to produce round numbers at the end of each epoch, so that TensorBoard
            # learning rate graphs look better.
            epoch += 1. / self.steps_per_epoch
            return 1. / mc.get_nranks() * (epoch * (mc.get_nranks() - 1) / warmup_epochs + 1)
            
        super(LearningRateWarmupCallback, self).__init__(
            multiplier, start_epoch=0, end_epoch=warmup_epochs, staircase=False,
            momentum_correction=momentum_correction, steps_per_epoch=steps_per_epoch, *args)
        self.verbose = verbose
        

    def on_epoch_end(self, epoch, logs=None):
        super(LearningRateWarmupCallback, self).on_epoch_end(epoch, logs)

        if epoch == self.end_epoch - 1 and self.verbose > 0:
            new_lr = self.backend.get_value(self.model.optimizer.lr)
            print('\nEpoch %d: finished gradual learning rate warmup to %g.' % (epoch + 1, new_lr))
