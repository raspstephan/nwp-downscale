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
    from run_src.dataloader import TiggeMRMSDataset
#     from run_src.utils import *
    from src.evaluation import par_gen_patch_eval, gen_patch_eval

    #set seed
    torch.manual_seed(args.seed)

    ## Load Data and set data params
    ds_test = pickle.load(open(args.data_hparams["test_dataset_path"], "rb"))  
    sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.eval_hparams["batch_size"], sampler=sampler_test
    )

    print("Loading data ... ")

    gan = GANs[args.gan].load_from_checkpoint(args.model_path)
    
    print("loaded gan")
    
    gen = gan.gen
    gen = gen.to(device)
    gen.train(False);

    print("Data loading complete")

    ## Load Model

    metrics = par_gen_patch_eval(gen, dl_test, args.eval_hparams["num_ens"], ds_test.mins.tp.values, ds_test.maxs.tp.values, ds_test.tp_log, device)    

    print(metrics)

    pickle.dump(metrics, open(model_dir+"/eval_metrics.pkl", "wb"))

if __name__ == '__main__':
    evaluate(input_args)