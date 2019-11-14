#!/bin/bash
#SBATCH -J hpo-cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

#config=configs/hpo_cifar10_cnn.yaml
script=hpo_train.py

srun python $script $config
