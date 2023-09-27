#!/usr/bin/env bash -l

conda update conda
conda create -n hisup python=3.7

conda init bash

conda run -n hisup conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch
conda run -n hisup conda install pycocotools=2.0.4 cudatoolkit-dev=11.7 -c conda-forge

conda run -n hisup conda develop .
conda run -n hisup pip install -r requirements.txt

# Should really point to /usr/local/cuda, however nvcc cuda compiler is not found during build afterwards.
export CUDA_HOME=/opt/conda/envs/hisup

# conda run -n hisup make ### Instead of make just copy its contents
cd /storage/experiments/hisup/hisup/csrc/lib/afm_op; python setup.py build_ext --inplace; rm -rf build
cd /storage/experiments/hisup/hisup/csrc/lib/squeeze; python setup.py build_ext --inplace; rm -rf build

cd /storage/experiments
