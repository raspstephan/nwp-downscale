import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from catalyst.data.sampler import DistributedSamplerWrapper

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pickle

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")
    
import json
import argparse
import sys

def parseInputArgs():

    parser = argparse.ArgumentParser(
                description="specify_experiment_parameters",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--eval_config", type=str,
                        dest="config_path", help='Path to file containing configuration for the evaluation')
    
    parser.add_argument("--type", type=str, dest="eval_type", default="full_field")

    return parser.parse_args()

input_args = parseInputArgs()

def evaluate(input_args):
    # Load Experiment Args and Hyperparameters

    args = json.load(open(input_args.config_path))
    parser = argparse.ArgumentParser(args)
    parser.set_defaults(**args)
    args, _ = parser.parse_known_args()

    print("Args loaded")
    
    device = torch.device(f'cuda:{args.gpu}')
    print("device", device)
    model_dir = args.save_hparams["save_dir"]+args.save_hparams["run_name"]+str(args.save_hparams["run_number"])+"/"

    sys.path.append(model_dir)

    from run_src.models import GANs
    from src.dataloader import TiggeMRMSPatchLoadDataset
#     from run_src.utils import *
    from src.evaluation import par_gen_patch_eval, gen_patch_eval, par_gen_full_field_eval, par_SR_gen_patch_eval, par_gen_patch_remi_eval

    #set seed
    torch.manual_seed(args.seed)

    ## Load Data and set data params
    ds_test = TiggeMRMSPatchLoadDataset(args.data_hparams["test_dataset_path"], samples_vars=args.data_hparams['samples_vars'])  
#     test_batch_idxs = np.load("/home/jupyter/data/data_patches/test/configs/test_batch_idxs.npy", allow_pickle=True)
#     ds_test = torch.utils.data.Subset(ds_test, test_batch_idxs)
    
    sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.eval_hparams["batch_size"], sampler=sampler_test
    )

    test_args = pickle.load(open(args.data_hparams['test_dataset_path']+'/configs/dataset_args.pkl', 'rb'))
    
    print("Loading data ... ")

    gan = GANs[args.gan].load_from_checkpoint(args.model_path)
    
    print("loaded gan")
    
    gen = gan.gen
    gen = gen.to(device)
    gen.train(False);

    print("Data loading complete")
#     print("ds test type", type(ds_test))
    ## Load Model
    if input_args.eval_type == "patch":
        if 'super_resolution' in args:
            metrics = par_SR_gen_patch_eval(gen, dl_test, args.eval_hparams["num_ens"], test_args['mins'].tp.values, test_args['maxs'].tp.values, test_args['tp_log'], device)
        elif 'remi' in args:
            metrics = par_gen_patch_remi_eval(gen, dl_test, args.eval_hparams["num_ens"], test_args['mins'].tp.values, test_args['maxs'].tp.values, test_args['tp_log'], device)       
        else:
            metrics = par_gen_patch_eval(gen, dl_test, args.eval_hparams["num_ens"], test_args['mins'].tp.values, test_args['maxs'].tp.values, test_args['tp_log'], device)    
    elif input_args.eval_type == "full_field":
        metrics = par_gen_full_field_eval(gen, ds_test, args.eval_hparams["num_ens"], test_args['mins'].tp.values, test_args['maxs'].tp.values, test_args['tp_log'], device)
        
    print(metrics)

    pickle.dump(metrics, open(model_dir+f"/{input_args.eval_type}_eval_metrics_new.pkl", "wb"))

if __name__ == '__main__':
    evaluate(input_args)