# Hello World Examples

This directory contains a "hello world" of hyperparameter optimization:
Find the coefficients (our hyperparameter) of a 5th order polynomial that
best fit a sin wave between -π and π.

The sin example is a nice way to introduce the HPO interface due to the quick
computation time.

## Kernel Script

Take a look at the kernel script. See how the hyperparameters (polynomial
coefficients) are exposed as command line options through argparse and how the
figure of merit (cost function) is exposed through a print statement.

If you want to run this script on the login node, be sure to load python3
through the tensorflow module if you have not already:

    module load tensorflow

The kernel script can be run via command line on the login node:

    # From this directory
    python3 source/sin.py

Try running it directly with the default arguments. You are welcome to take a
try hand-tuning the hyperparameters to lower the FoM value before trying HPO.
The script supports a `--help` flag which lists the options:

    # From this directory
    python3 source/sin.py --help


## HPO Driver

Take a look at each of the hpo examples, and see how the hyperparameter
optimization is set up. Each example includes some comments describing the
interface. You can also refer to the API documentation for reference:

https://cray.github.io/crayai/hpo/hpo.html

The HPO scripts should be launched onto an allocation by submitting their
respective batch script:

    # From top-level directory of repository
    sbatch scripts/hpo_grid_example.sh
    sbatch scripts/hpo_random_example.sh
    sbatch scripts/hpo_genetic_example.sh

Each of these examples takes 30-60 seconds to run to completion on 1 node.
