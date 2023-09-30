#!/usr/bin/env bash

conda update conda
conda create -n hisup python=3.7

conda init bash

conda install -n hisup pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch
conda install -n hisup pycocotools=2.0.4 cudatoolkit-dev -c conda-forge

conda run -n hisup conda develop .
conda run -n hisup pip install -r requirements.txt

conda list --explicit > /storage/experiments/hisup/conda_requirements.txt
conda env export -n hisup > /storage/experiments/hisup/environment.yaml

# find / -name nvcc >> /storage/experiments/hisup/nvcc_path.txt

# Should really point to /usr/local/cuda, however nvcc cuda compiler is not found during build afterwards.
