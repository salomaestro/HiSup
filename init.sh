#!/bin/sh

# Conda requirements install script
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
#
# bash ~/miniconda.sh -b -p /opt/conda
#
# ls /opt/conda

# export PATH="/root/miniconda/bin:${PATH}"
#
conda update conda
conda create -n hisup python=3.7

conda init bash
conda activate hisup

conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch
conda install pycocotools=2.0.4 -c conda-forge

conda develop .
pip install -r requirements.txt

cd /experiment/hisup/csrc/lib
make

cd /experiment
# export PATH="/root/miniconda/envs/hisup/bin:${PATH}"
