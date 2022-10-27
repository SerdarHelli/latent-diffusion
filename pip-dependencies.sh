#!/bin/sh
. $PREFIX/etc/profile.d/conda.sh  # do not edit
conda activate $PREFIX            # do not edit

# add your pip packages here; bottle is just an example!
# replace it with your dependencies
python -m pip install bottle

python -m pip install  albumentations==0.4.3
python -m pip install  opencv-python==4.1.2.30
python -m pip install pudb==2019.2
python -m pip install  imageio==2.9.0
python -m pip install  imageio-ffmpeg==0.4.2
python -m pip install  pytorch-lightning==1.4.2
python -m pip install  omegaconf==2.1.1
python -m pip install  test-tube>=0.7.5
python -m pip install  streamlit>=0.73.1
python -m pip install  einops==0.3.0
python -m pip install  torch-fidelity==0.3.0
python -m pip install  transformers==4.3.1
python -m pip install  -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
python -m pip install  -e git+https://github.com/openai/CLIP.git@main#egg=clip
python -m pip install  torchvision==0.8.1
python -m pip install  natsort==8.2.0
python -m pip install  shutils==0.1.0
python -m pip install  torchmetrics==0.6.0
python -m pip install  kornia==0.6.8
