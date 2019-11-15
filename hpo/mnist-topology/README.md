# MNIST Topology Optimization Example

This directory contains an HPO example of doing a topology optimization.

## Kernel Script

The kernel script (source/mnist.py) trains the LeNet-5 CNN to predict the
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

This example should take approximately 10 minutes to run to completion.
