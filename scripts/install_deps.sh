#!/bin/bash
conda create -n redepth python=3.12
conda activate redepth

pip install \
    torch \
    torchvision \
    omegaconf \
    tensorboard \
    matplotlib \
    diffusers==0.35.1 \
    transformers==4.56.1 \
    accelerate \
    opencv-python \
    bitsandbytes \
    tqdm








