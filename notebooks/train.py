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

from src.models import GANs, gens, discs
# from ilan_src.models import *
from src.dataloader import *
from src.utils import *

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



def parseInputArgs():

    parser = argparse.ArgumentParser(
                description="specify_experiment_parameters",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--experiment_config", type=str,
                        dest="config_path", help='Path to file containing configuration for the experiment')

    return parser.parse_args()

input_args = parseInputArgs()


def train(input_args):
    # Load Experiment Args and Hyperparameters

    args = json.load(open(input_args.config_path))
    parser = argparse.ArgumentParser(args)
    parser.set_defaults(**args)
    args, _ = parser.parse_known_args()

    print("Args loaded")
    # set seed
    torch.manual_seed(args.seed)
          
    args.gan_hparams['generator'] = gens[args.gan_hparams['generator']]
    args.gan_hparams['discriminator'] = discs[args.gan_hparams['discriminator']]

    ## Load Data and set data params
    
    print("Loading data ... ")
    ds_train = pickle.load(open(args.data_hparams['train_dataset_path'], "rb"))
    ds_valid = pickle.load(open(args.data_hparams['valid_dataset_path'], "rb"))

    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.compute_weights(), len(ds_train))
    sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas = args.train_hparams['gpus'], rank = 0)
    sampler_valid = torch.utils.data.WeightedRandomSampler(ds_valid.compute_weights(), len(ds_valid))
    sampler_valid = DistributedSamplerWrapper(sampler_valid, num_replicas = args.train_hparams['gpus'], rank = 0)
    
    batch_size = args.train_hparams['batch_size']//args.train_hparams['gpus']
    
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=6)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=6)

    
    args.gan_hparams['val_hparams']['ds_max'] = ds_train.maxs.tp.values
    args.gan_hparams['val_hparams']['ds_min'] = ds_train.mins.tp.values
    args.gan_hparams['val_hparams']['tp_log'] = ds_train.tp_log

    del ds_train
    del ds_valid
    
    print("Data loading complete")
    ## Load Model

    model = GANs[args.gan](**args.gan_hparams)

    ## Define trainer and logging

    save_dir = args.save_hparams['save_dir']

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir+args.save_hparams['run_name']+str(args.save_hparams['run_number']) + '/')
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = '../logs/',
                                             name = args.save_hparams['run_name'], 
                                             version = args.save_hparams['run_number'])


    trainer = pl.Trainer(accelerator='ddp', 
                         precision=16, gpus = args.train_hparams['gpus'], 
                         max_epochs = args.train_hparams['epochs'], 
                         callbacks=[checkpoint_callback], 
                         replace_sampler_ddp = False, 
                         check_val_every_n_epoch=20, 
                         logger = tb_logger
                        )
                         
    print("Training model...")
    
    # Train
    trainer.fit(model, dl_train, dl_valid)    
    

if __name__ == '__main__':

    train(input_args)
    