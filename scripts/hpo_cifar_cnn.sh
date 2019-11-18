#!/bin/bash
#SBATCH -J hpo-cifar-cnn
#SBATCH --reservation dl4sci_sc19
#SBATCH -C knl
#SBATCH -N 8
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

# OpenMP settings
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"

script=hpo_train.py
args="-N ${SLURM_JOB_NUM_NODES} --verbose"

# Ensure dataset is downloaded by single process

echo "python -c \"import keras; keras.datasets.cifar10.load_data()\""
python -c "import keras; keras.datasets.cifar10.load_data()"

echo "python -u $scripts $args"
python -u $script $args
