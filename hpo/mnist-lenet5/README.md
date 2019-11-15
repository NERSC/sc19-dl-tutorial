# LeNet-5 trained on MNIST digits dataset

This directory contains an HPO example of doing a topology optimization.
This is an interesting example for 2 reasons:

1. The kernel script has been modified to minimize elapsed time instead of loss.
2. The hyperparameters and network topology are being optimized simultaneously.

There is a reasonable chance you will not have time to run this example to
completion within the time frame of the tutorial, and that is OK. Running this
example is not required to understand what this section aims to demonstrate.

## LeNet and MNIST digits dataset

The LeNet architecture is a CNN designed by Yann LeCunn for the task of image
classification of handwritten digit images.

It consists of two convolutional layers, each of which is followed by a
subsampling layer, and then a pair of fully-connected layers with
a final output layer.

The MNIST dataset contains 70,000 labeled images of handwritten numerals; each
single channel, 28×28pixel resolution. Ten thousand images are held out to form
a testing set.

## Kernel Script

The kernel script (`source/mnist.py`) trains the LeNet-5 CNN to predict the
values of handwritten digits from 0 to 9 from the MNIST digits dataset.
This script originates from the tensorflow repository and was modified to
expose the hyperparameters as command line arguments and print out the figure
of merit.

The exposed hyperparameters include:

- `dropout`:    Dropout rate used for generalization
- `momentum`:   Momentum factor
- `c1\_sz`:     Filter size of the first convolutional layer
- `c1\_ft`:     Filter count of the first convolutional layer
- `c2\_sz`:     Filter size of the second convolutional layer
- `c2\_ft`:     Filter count of the second convolutional layer
- `fullyc\_sz`: Width of the fully connected layer

To run the kernel script directly, use the batch script:

    # From top-level directory of repository
    sbatch scripts/mnist_topo.sh

A single run should take about 1 minute to complete.

## HPO Driver

To run the HPO driver, use the batch script:

    # From top-level directory of repository
    sbatch scripts/mnist_topo.sh

This example should take ~12 minutes to run to completion.
