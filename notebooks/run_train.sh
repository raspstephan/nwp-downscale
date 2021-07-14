#!/bin/bash

constants=$(cat ${1} | jq '.save_hparams' | jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]")

for key in ${constants}; do
  eval ${key}
done

python train.py --experiment_config $1

dest="${save_dir}${run_name}${run_number}/run_src"
folder="../src"

cp -r $folder $dest