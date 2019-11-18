#!/bin/bash
#SBATCH -J hpo-mnist-lenet5
#SBATCH --reservation dl4sci_sc19
#SBATCH -C knl
#SBATCH -N 4
#SBATCH -q debug
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

# OpenMP settings
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"

script=genetic.py
args="-N ${SLURM_JOB_NUM_NODES} --verbose"
path=hpo/mnist-lenet5

echo "cd $path && python -u $scripts $args"
cd $path && python -u $script $args
