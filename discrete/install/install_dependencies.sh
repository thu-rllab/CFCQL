#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.8 -y
# conda activate pymarl

# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
pip install sacred matplotlib tensorboard-logger libtmux h5py pyyaml  gym opencv-python

pip install protobuf==3.20.*

# pip install git+https://github.com/oxwhirl/smac.git
