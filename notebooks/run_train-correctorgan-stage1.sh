#!/bin/bash

constants=$(cat ${1} | jq '.save_hparams' | jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]")

for key in ${constants}; do
  eval ${key}
done

if [ $# -eq 2 ]
then
    dest="${save_dir}${run_name}${run_number}/"
    
    if [ ! -d "dest" ]; then
        mkdir -p "${dest}"

        folder="../src"

        echo $dest

        \cp -r $folder $dest

        rm -r "${dest}run_src" 

        mv "${dest}src" "${dest}run_src"
    fi
    
    python train-correctorgan_stage1.py --experiment_config $1 --ckpt_path $2

else
    dest="${save_dir}${run_name}${run_number}/"

    mkdir -p "${dest}"

    folder="../src"
    
    echo $dest

    \cp -r $folder $dest
    
    rm -r "${dest}run_src" 
    
    mv "${dest}src" "${dest}run_src" 

    python train-correctorgan_stage1.py --experiment_config $1
fi