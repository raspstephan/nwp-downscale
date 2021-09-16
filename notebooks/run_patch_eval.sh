#!/bin/bash

# loop over model number args

for MODEL in "$@"
do
    filepath=$(jq --arg m $MODEL '.[$m]' /home/jupyter/nwp-downscale/experiments/model_eval_config_paths.json )
    filepath="${filepath%\"}"
    filepath="${filepath#\"}"
    python eval-load-images.py --eval_config $filepath --type patch &
    python visualise_predictions-load-images.py --eval_config $filepath &
done