#!/usr/bin/env bash

# conda create -n hisup --file /storage/experiments/hisup/conda_requirements.txt
conda env create --file /storage/experiments/hisup/environment.yaml

echo "=========================== ENV CREATED ==========================="

which conda

# export CUDA_HOME=/opt/conda/envs/hisup/pkgs/cuda-toolkit
export CUDA_HOME=/opt/conda/envs/hisup

# Add hisup bin to PATH
export PATH=/storage/experiments/hisup/hisup/bin:$PATH

echo $PATH

echo "TEST CUDA HOME: $CUDA_HOME"
ls $CUDA_HOME

echo "=========================== CONTINUING BUILD ==========================="

# conda run -n hisup make ### Instead of make just copy its contents
cd /storage/experiments/hisup/hisup/csrc/lib/afm_op; conda run -n hisup python setup.py build_ext --inplace; rm -rf build
echo "=========================== AFM_OP DONE ==========================="
cd /storage/experiments/hisup/hisup/csrc/lib/squeeze; conda run -n hisup python setup.py build_ext --inplace; rm -rf build

echo "=========================== SQUEEZE DONE ==========================="

cd /storage/experiments
