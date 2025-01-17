FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
MAINTAINER csalomonsen <christian.salomonsen@uit.no>

WORKDIR /storage/experiments/hisup

COPY requirements.txt requirements.txt
COPY docker/docker-entrypoint.sh /home

ENV CONDA_ALWAYS_YES="true"
ENV DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
ENV CUDA_HOME=/usr/local/cuda

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y wget curl gcc make git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install --reinstall build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
RUN git clone https://github.com/bowenc0221/boundary-iou-api.git
RUN pip install -e boundary-iou-api

ENTRYPOINT ["/home/docker-entrypoint.sh"]
