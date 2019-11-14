#!/bin/bash
#SBATCH -J cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36

config=configs/cifar10_cnn.yaml
script=train.py

srun python $script $config -d
