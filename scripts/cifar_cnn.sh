#!/bin/bash
#SBATCH -J cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

##SBATCH --reservation dl4sci_sc19

# Load the software
module load tensorflow/intel-1.13.1-py36
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
config=configs/cifar10_cnn.yaml

# Ensure dataset is downloaded by single process
python -c "import keras; keras.datasets.cifar10.load_data()"

# Run the training
srun python train.py $config -d
