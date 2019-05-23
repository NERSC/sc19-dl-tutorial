#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q regular
#SBATCH --reservation=cug_analytics_2019
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. /usr/common/software/python/3.6-anaconda-5.2/etc/profile.d/conda.sh
conda activate /global/cscratch1/sd/sfarrell/conda/cug19
module use /global/cscratch1/sd/kristyn/Urika-XC1.2/opt/cray/pe/modulefiles
module load craype-ml-plugin-py3/gnu71/1.1.4

config=configs/cifar10_resnet.yaml
script=train_cpe_ml.py

srun python $script $config -d
