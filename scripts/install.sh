#!/bin/bash

# Example script showing how we installed TensorFlow

# Create conda env
. /usr/common/software/python/3.6-anaconda-5.2/etc/profile.d/conda.sh
INSTALL_PATH=$SCRATCH/conda/cug19
conda create -p $INSTALL_PATH -y python=3.6 numpy ipykernel matplotlib

# Install TF + Keras
conda activate $INSTALL_PATH
pip install https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
pip install keras
