#!/bin/bash
#SBATCH -J cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

##SBATCH --reservation dl4sci_sc19

module load tensorflow/intel-1.13.1-py36
config=configs/cifar10_cnn.yaml

srun python train.py $config -d
