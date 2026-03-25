FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG CONDA_PREFIX=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git vim bzip2 ca-certificates \
        build-essential cmake ninja-build \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
        libopencv-dev libeigen3-dev \
        libgoogle-glog-dev libatlas-base-dev \
        ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
        libbullet-dev \
        lsb-release gnupg2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://repo.anaconda.com/miniconda/${CONDA_VERSION}.sh" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_PREFIX} \
    && rm /tmp/miniconda.sh \
    && ${CONDA_PREFIX}/bin/conda init bash \
    && ${CONDA_PREFIX}/bin/conda config --set always_yes true \
    && ${CONDA_PREFIX}/bin/conda config --append channels conda-forge \
    && ${CONDA_PREFIX}/bin/conda config --append channels aihabitat \
    && ${CONDA_PREFIX}/bin/conda clean -afy

ENV PATH="${CONDA_PREFIX}/bin:${PATH}"

RUN conda create -n vln_ce python=3.9 -c conda-forge \
    && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "vln_ce", "/bin/bash", "-c"]

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu121
RUN pip install \
        opencv-python-headless==4.12.0 \
        Pillow==10.4.0 \
        imageio==2.37.2 \
        imageio-ffmpeg==0.6.0 \
        moviepy==2.2.1 \
        pycocotools==2.0.11 \
        av==15.1.0

RUN conda install -n vln_ce \
"habitat-sim=0.3.3" withbullet headless \
-c aihabitat -c conda-forge \
&& conda clean -afy

RUN pip install plotly==6.5.2
RUN pip install ultralytics==8.3.181

RUN echo "conda activate vln_ce" >> /root/.bashrc

WORKDIR /workspace
RUN git clone https://github.com/MlLearnerAkash/AM_vln_ce.git
WORKDIR /workspace/AM_vln_ce

# RUN python download_mp3d.py --task habitat -o data/scene_datasets/mp3d/
