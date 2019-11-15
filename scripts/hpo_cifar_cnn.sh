#!/bin/bash
#SBATCH -J hpo-cifar-cnn
#SBATCH -C knl
#SBATCH -N 8
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

##SBATCH --reservation dl4sci_sc19

module load tensorflow/intel-1.13.1-py36
module load cray-hpo

export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"

script=hpo_train.py
args="-N ${SLURM_JOB_NUM_NODES} --verbose"

# Ensure dataset is downloaded by single process
python -c "import keras; keras.datasets.cifar10.load_data()"

python $script $args
