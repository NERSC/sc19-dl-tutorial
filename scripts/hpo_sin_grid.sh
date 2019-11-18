#!/bin/bash
#SBATCH -J hpo-sin-grid
#SBATCH --reservation dl4sci_sc19
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 5
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=grid_example.py
path=hpo/sin

cd $path && python -u $script
