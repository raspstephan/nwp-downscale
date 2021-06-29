#!/bin/bash

conda env create --file ilan_env.yml
conda activate ilan
conda install pip
conda deactivate
conda deactivate
conda activate ilan
pip install -r ilan_env_requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html



