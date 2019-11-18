# SC19 Tutorial: Deep Learning At Scale

This repository contains the material for the SC19 tutorial:
*Deep Learning at Scale*.

Here you will links to slides and resources as well as all the code
for the hands-on sessions.
It contains specifications for a few datasets, a couple of CNN models, and
all the training code to enable training the models in a distributed fashion
using Horovod.

As part of the tutorial, you will
1. Train a simple CNN to classify images from the CIFAR10 dataset on a single node
2. Train a ResNet model to classify the same images on multiple nodes

**Contents**
* [Links](https://github.com/NERSC/sc19-dl-tutorial#links)
* [Installation](https://github.com/NERSC/sc19-dl-tutorial#installation)
* [Navigating the repository](https://github.com/NERSC/sc19-dl-tutorial#navigating-the-repository)
* [Hands-on walk-through](https://github.com/NERSC/sc19-dl-tutorial#hands-on-walk-through)
    * [Single node training example](https://github.com/NERSC/sc19-dl-tutorial#single-node-training-example)
    * [Multi-node training example](https://github.com/NERSC/sc19-dl-tutorial#multi-node-training-example)
    * [Advanced example: multi-node ResNet50 on ImageNet-100](https://github.com/NERSC/sc19-dl-tutorial#advanced-example-multi-node-resnet50-on-imagenet-100)
* [Code references](https://github.com/NERSC/sc19-dl-tutorial#code-references)

## Links

Presentation slides: https://drive.google.com/drive/folders/1KJm08Ry4qJXOl19MAu2Ao1t_fRNaMwZn?usp=sharing

NERSC JupyterHub: https://jupyter.nersc.gov

Join Slack: https://join.slack.com/t/nersc-dl-tutorial/shared_invite/enQtODMzMzQ1MTI5OTUyLWNlNzg2MjBkODIwODRlNTBkM2M4MjI0ZDk2ZDU4N2M3NjU5MDk1NTRmMTFhNWRkMTk0NGNhMzQ3YjU2NzU5NTk

## Installation

1. Start a terminal on Cori, either via ssh or from the Jupyter interface.
2. Clone the repository using git:\
   `git clone https://github.com/NERSC/sc19-dl-tutorial.git`

That's it! The rest of the software (Keras, TensorFlow) is pre-installed on Cori
and loaded via the scripts used below.

## Navigating the repository

**`train.py`** - the main training script which can be steered with YAML
configuration files.

**`data/`** - folder containing the specifications of the datasets. Each dataset
has a corresponding name which is mapped to the specification in `data/__init__.py`

**`models/`** - folder containing the Keras model definitions. Again, each model
has a name which is interpreted in `models/__init__.py`.

**`configs/`** - folder containing the configuration files. Each
configuration specifies a dataset, a model, and all relevant configuration
options (with some exceptions like the number of nodes, which is specified
instead to SLURM via the command line).

**`scripts/`** - contains an environment setup script and some SLURM scripts
for easily submitting the example jobs to the Cori batch system.

**`utils/`** - contains additional useful code for the training script, e.g.
custom callbacks, device configuration, and optimizers logic.

**`hpo/`** - contains READMEs and examples for HPO hands-on.

## Hands-on walk-through

Go through the following steps as directed by the tutorial presenters.
Discuss the questions with your neighbors.

### Single node training example

We will start with single node training of a simple CNN to classify images
from the CIFAR10 dataset.

1. Take a look at the simple CNN model defined here: [models/cnn.py](models/cnn.py).
   Consider the following things:
    * Note how the model is constructed as a sequence of layers
    * Note the structure of alternating convolutions, pooling, and dropout
    * Identify the _classifier head_ of the model; the part which computes the
      class probabilities.
    * *Can you figure out what the `Flatten()` layer does here,
      and why it is needed?*

2. Now take a look at the dataset code for CIFAR10: [data/cifar10.py](data/cifar10.py)
    * Keras has a convenient API for CIFAR10 which will automatically download
      the dataset for you.
    * Ask yourself: *why do we scale the dataset by 1/255?*
    * Note where we convert the labels (integers) to categorical class vectors.
      Ask yourself: *why do we have to do this?*
    * *What kinds of data augmentation are we applying?*

3. Next, take a look at the training script: [train.py](train.py).
    * Identify the part where we retrieve the dataset.
    * Identify the section where we retrieve the CNN model, the optimizer, and
      compile the model.
    * Now identify the part where we do the actual training.

4. Finally, look at the configuration file: [configs/cifar10_cnn.yaml](configs/cifar10_cnn.yaml).
    * YAML allows to express configurations in rich, human-readable, hierarchical structure.
    * Identify where you would edit to modify the optimizer, learning-rate, batch-size, etc.

5. Now we are ready to submit our training job to the Cori batch system.
   We have provided SLURM scripts to make this as simple as possible.
   To run the simple CNN training on CIFAR10 on a single KNL node, simply do:\
   `sbatch scripts/cifar_cnn.sh`
    * **Important:** the first time you run a CIFAR10 example, it will
    automatically download the dataset. If you have more than one job attempting
    this download simultaneously it will likely fail.

6. Check on the status of your job by running `sqs`.
   Once the job starts running, you should see the output start to appear in the
   slurm log file `logs/cifar-cnn-*.out`.

7. When the job is finished, check the log to identify how well your model learned
   to solve the CIFAR10 classification task. For every epoch you should see the
   loss and accuracy reported for both the training set and the validation set.
   Take note of the best validation accuracy achieved.

### Multi-node training example

To demonstrate scaling to multiple nodes, we will switch to a larger, more complex
ResNet model. This model can achieve higher accuracy than our simple CNN, but
it is quite a bit slower to train. By parallelizing the training across nodes
we should be able to achieve a better result than our simple CNN in a practical
amount of time.

1. Check out the ResNet model code in [models/resnet.py](models/resnet.py).
   Note this is quite a bit more complex than the simple CNN! In fact the model
   code is broken into multiple functions for easy reuse. We provide here two
   versions of ResNet models: a standard ResNet50 (with 50 layers) and a smaller
   ResNet consisting of 26 layers.
    * Identify the identy block and conv block functions. *How many convolutional
      layers do each of these have*?
    * Identify the functions that build the ResNet50 and the ResNetSmall. Given how
      many layers are in each block, *see if you can confirm how many layers (conv
      and dense) are in the models*. **Hint:** we don't normally count the
      convolution applied to the shortcuts.

2. Inspect the optimizer setup in [utils/optimizers.py](utils/optimizers.py).
    * Note how we scale the learning rate (`lr`) according to the number of
      processes (ranks).
    * Note how we construct our optimizer and then wrap it in the Horovod
      DistributedOptimizer.

3. Inspect [train.py](train.py) once again.
    * Identify the `init_workers` function where we initialize Horovod.
      Note where this is invoked in the main() function (right away).
    * Identify where we setup our training callbacks.
    * *Which callback ensures we have consistent model weights at the start of training?*
    * Identify the callbacks responsible for the learning rate schedule (warmup and decay).

That's mostly it for the code. Note that in general when training distributed
you might want to use more complicated data handling, e.g. to ensure different
workers are always processing different samples of your data within a training
epoch. In this case we aren't worrying about that and are, for simplicity,
relying on the independent random shuffling of the data by each worker as well
as the random data augmentation.

4. (**optional**) To gain an appreciation for the speedup of training on
   multiple nodes, you can first try to train the ResNet model on a single node.
   Adjust the configuration in [configs/cifar10_resnet.yaml](configs/cifar10_resnet.yaml)
   to train for just 1 epoch and then submit the job with\
   `sbatch -N 1 scripts/cifar_resnet.sh`

5. Now we are ready to train our ResNet model on multiple nodes using Horovod
   and MPI! If you changed the config to 1 epoch above, be sure to change it back
   to 32 epochs for this step. To launch the ResNet training on 8 nodes, do:\
   `sbatch -N 8 scripts/cifar_resnet.sh`

6. As before, watch the log file (`logs/cifar-resnet-*.out`) when the job starts.
   You'll see some printouts from every worker. Others are only printed from rank 0.

7. When the job is finished, look at the log and compare to the simple CNN case
   above. If you ran step 4, compare the time to train one epoch between single-node
   and multi-node. *Did your model manage to converge to a better validation accuracy
   than the simple CNN?*

Now that you've finished the main tutorial material, try to play with the code
and/or configuration to see the effect on the training results. You can try changing
things like
* Change the optimizer (search for Keras optimizers on google).
* Change the nominal learning rate, number of warmup epochs, decay schedule
* Change the learning rate scaling (e.g. try "sqrt" scaling instead of linear)

Most of these things can be changed entirely within the configuration.
See [configs/imagenet_resnet.yaml](configs/imagenet_resnet.yaml) for examples.

### Hyperparameter Optimization

The following examples will walk you through how to utilize distributed
hyperparameter optimization with the Cray HPO package.

For documentation reference, see the
[Cray HPO documentation](https://cray.github.io/crayai/hpo/hpo.html).

#### Example: Hello World HPO examples

This is a set of quick-running examples that you may view and run to get
acquainted with the Cray HPO interface.

Further instructions are in: `hpo/sin/README.md`

#### Example: Applying distributed HPO to CNN CIFAR10 example

In this example, we will be optimizing the hyperparameters of the CNN single node
training example from before.

1. Take a look `train.py` again and follow the `--hpo` argument. Note that the
   loss value is being emitted when this argument is set, which is necessary to
   communicate the figure of merit back to the Cray HPO optimizer. Additionally,
   inspect the `configs/hpo_cifar10_cnn.yaml` configuration, and note that the
   number of epochs has been scaled down significantly so that this example can
   run to completion in a reasonable amount of time.

2. Take a look at the `hpo_train.py` HPO driver. This script sets up the
   evaluator, hyperparameter search space, and optimizer. Make sure you
   understand the HPO code by trying to answer these questions:

   * What hyperparameters are being optimized?
   * Which optimizer is being used for this example?
   * How many evaluations of `train.py` will this optimization run?

3. Now we are ready to run our hyperparameter optimization. Similar to before,
   a SLURM script is provided to run the HPO driver on 8 KNL nodes:

   ```
   sbatch scripts/hpo_cifar_cnn.sh
   ```

   If no modifications were made, this should take ~12 minutes to run to
   completion.

4. Take a look at your job output file (`*.out`) in the `logs/` directory. Try
   to identify the following:

    * How much was your figure of merit improved?
    * What were the improved hyperparameter values found?


#### Example: Optimizing topology of LeNet-5

This is an example of using Cray HPO to optimize hyperparameters for LeNet-5
trained on the MNIST hand-written digits dataset.

This example is unique because the figure of merit is the elapsed training time
until a threshold accuracy is reached, to minimize time-to-accuracy.
Additionally, this example shows how one can optimize network topology with
other traditional hyperparameters.

Further instructions are in: `hpo/mnist-lenet5/README.md`

## Code references

Keras ResNet50 official model:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

Horovod ResNet + ImageNet example:
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py

CIFAR10 CNN and ResNet examples:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
