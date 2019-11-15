#!/bin/bash
#SBATCH -J hpo-sin-random
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=random_example.py
path=hpo/sin

cd $path && srun python $script
