# BASE IMAGE
FROM continuumio/miniconda3

SHELL ["/bin/bash","-c"]

WORKDIR /opt

RUN apt-get update && apt-get -y install git gcc g++ liblapack-dev libblas-dev libboost-dev libarmadillo-dev libfftw3-dev \
    && git clone https://github.com/xulabs/aitom.git && cd aitom \
    && pip install cython numpy==1.19.2 \
    && bash build.sh

EXPOSE 8888

