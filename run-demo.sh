#!/bin/bash

python -V

# export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
# export CUDA_HOME=/usr/local/cuda
#
# echo "=========================== STARTING INSTALL ==========================="
#
# apt-get update && apt-get install -y ffmpeg libsm6 libxext6 gcc make
# apt-get update && apt-get install -y --reinstall build-essential
#
# echo "=========================== STARTING PIP INSTALL ==========================="
#
#
# # Install dependencies
# pip install -r requirements2.txt
#
# echo "=========================== STARTING BUILD ==========================="
#
# # conda run -n hisup make ### Instead of make just copy its contents
# cd /storage/experiments/hisup/hisup/csrc/lib/afm_op; python setup.py build_ext --inplace; rm -rf build
# echo "=========================== AFM_OP DONE ==========================="
# cd /storage/experiments/hisup/hisup/csrc/lib/squeeze; python setup.py build_ext --inplace; rm -rf build
#
# echo "=========================== SQUEEZE DONE ==========================="
#
# cd /storage/experiments/hisup
#
# echo "=========================== STARTING DEMO ==========================="
#
# python3 scripts/demo.py --dataset crowdai --img 000000000027.jpg
#
# echo "=========================== DEMO DONE ==========================="
