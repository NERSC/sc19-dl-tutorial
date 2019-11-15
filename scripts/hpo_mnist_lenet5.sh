#!/bin/bash
#SBATCH -J hpo-mnist-lenet5
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -o logs/%x-%j.out

##SBATCH --reservation dl4sci_sc19

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

script=genetic.py
args="-N ${SLURM_JOB_NUM_NODES} --verbose"
path=hpo/mnist-lenet5

cd $path && python $script $args
