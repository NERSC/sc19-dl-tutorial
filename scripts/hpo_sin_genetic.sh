#!/bin/bash
#SBATCH -J hpo-sin-genetic
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 5
#SBATCH -o logs/%x-%j.out

##SBATCH --reservation dl4sci_sc19

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=genetic_example.py
path=hpo/sin

cd $path && python $script
