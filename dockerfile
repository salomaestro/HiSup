FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
MAINTAINER csalomonsen <christian.salomonsen@uit.no>

WORKDIR /storage/experiments/hisup

COPY . .

ENV CONDA_ALWAYS_YES="true"

# Install dependencies https://saturncloud.io/blog/how-to-install-packages-with-miniconda-in-dockerfile-a-guide-for-data-scientists/
RUN apt-get update && apt-get install -y wget gcc make git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install --reinstall build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN conda update conda

RUN conda env create --file /storage/experiments/hisup/environment.yaml

RUN conda init bash

SHELL ["conda", "run", "-n", "hisup", "/bin/bash", "-c"]

ENV CUDA_HOME=/opt/conda/envs/hisup

RUN conda run -n hisup conda develop .

# WORKDIR /storage/experiments/hisup/hisup/csrc/lib/afm_op
# RUN python setup.py build_ext --inplace
# RUN rm -rf build
#
# WORKDIR /storage/experiments/hisup/hisup/csrc/lib/squeeze
# RUN python setup.py build_ext --inplace
# RUN rm -rf build
#
# WORKDIR /storage/experiment/hisup
# RUN echo "source activate hisup" >> ~/.bashrc
# ENV PATH="/opt/conda/envs/hisup/bin:${PATH}"
