#!/bin/bash
#SBATCH -J mnist_topo
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=hpo/mnist-topology/source/mnist.py

srun python $script
