#!/bin/bash
#SBATCH -J hpo_sin_genetic
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=hpo/sin/genetic_example.py

srun python $script -d
