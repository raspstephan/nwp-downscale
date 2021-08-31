#!/bin/bash

conda env create --file ilan_env.yml
conda activate ilan
conda install pip
conda deactivate
conda deactivate
conda activate ilan
pip install -r ilan_env_requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

mkdir /opt/conda/envs/ilan/etc/conda/activate.d
mkdir /opt/conda/envs/ilan/etc/conda/deactivate.d
touch /opt/conda/envs/ilan/etc/conda/activate.d/env_vars.sh
touch /opt/conda/envs/ilan/etc/conda/deactivate.d/env_vars.sh

echo '#!/bin/sh' >> /opt/conda/envs/ilan/etc/conda/activate.d/env_vars.sh
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> /opt/conda/envs/ilan/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=/opt/conda/envs/ilan/lib/:${OLD_LD_LIBRARY_PATH}' >> /opt/conda/envs/ilan/etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' >> /opt/conda/envs/ilan/etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' >> /opt/conda/envs/ilan/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> /opt/conda/envs/ilan/etc/conda/deactivate.d/env_vars.sh

conda deactivate 
conda activate ilan
