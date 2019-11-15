#!/bin/bash
#SBATCH -J hpo-cifar-cnn
#SBATCH -C knl
#SBATCH -N 8
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=hpo_train.py
args=-N ${SLURM_JOB_NUM_NODES} --verbose

srun python $script $args
