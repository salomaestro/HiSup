FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
MAINTAINER csalomonsen <christian.salomonsen@uit.no>

WORKDIR /storage/experiments/hisup

ENV CONDA_ALWAYS_YES="true"

# Install dependencies https://saturncloud.io/blog/how-to-install-packages-with-miniconda-in-dockerfile-a-guide-for-data-scientists/
RUN apt-get update && apt-get install -y wget gcc make git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install --reinstall build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Should try to remove all below here, and instead use install script.

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# RUN bash ~/miniconda.sh -b -p $HOME/miniconda
#
# ENV PATH="/root/miniconda/bin:${PATH}"

# RUN conda update conda
#
# RUN conda create -n hisup python=3.7
#
# RUN conda init bash
#
# SHELL ["conda", "run", "-n", "hisup", "/bin/bash", "-c"]
#
# RUN conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch
# RUN conda install pycocotools=2.0.4 -c conda-forge
#
# RUN conda develop .
# RUN pip install -r requirements.txt
#
# # RUN conda install -c conda-forge opencv=4.5.5
#
# WORKDIR /experiment/hisup/csrc/lib
# RUN make
#
# WORKDIR /experiment
# RUN echo "source activate hisup" >> ~/.bashrc
# ENV PATH="/root/miniconda/envs/hisup/bin:${PATH}"
