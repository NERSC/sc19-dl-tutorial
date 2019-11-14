#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# BA: 1024 KNL nodes (~4 nodes/person?)
module load tensorflow/intel-1.13.1-py36

config=configs/cifar10_resnet.yaml
script=train_horovod.py

srun python $script $config -d
