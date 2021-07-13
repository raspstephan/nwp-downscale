## To install the environment:
1. 'conda init bash'
2. 'cd env_setup'
3. 'bash -i install_env.sh'
4. 'conda activate ilan'

## To train a model

1. Export the expriment configuration with the `export_experiment_args.ipynb' notebook in '/notebooks' to 'exp_path'
2. run './run_train exp_path'

## To evaluate a model
1. Export the evaluation configuration with the `export_experiment_args.ipynb' notebook in '/notebooks' to 'eval_path'
2. run 'python eval.py --eval_config eval_path'

## To create and save a dataset
1. Use the 'data-processing.ipynb' notebook in '/notebooks'

