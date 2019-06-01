#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH --reservation=cug_analytics_2019
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36

config=configs/cifar10_resnet.yaml
script=train_horovod.py

srun python $script $config -d
