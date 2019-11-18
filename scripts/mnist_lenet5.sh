#!/bin/bash
#SBATCH -J mnist-lenet5
#SBATCH --reservation dl4sci_sc19
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36

script=mnist.py
path=hpo/mnist-lenet5/source

cd $path && python $script
