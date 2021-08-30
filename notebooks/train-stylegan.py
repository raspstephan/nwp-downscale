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

    parser.add_argument("--experiment_config", type=str,
                        dest="config_path", help='Path to file containing configuration for the experiment')
    
    parser.add_argument("--ckpt_path", default = None, help= "path to checkpoint to continue training from")
    return parser.parse_args()

input_args = parseInputArgs()


def train(input_args):
    # Load Experiment Args and Hyperparameters

    args = json.load(open(input_args.config_path))
    parser = argparse.ArgumentParser(args)
    parser.set_defaults(**args)
    args, _ = parser.parse_known_args()
    
    model_dir = args.save_hparams["save_dir"]+args.save_hparams["run_name"]+str(args.save_hparams["run_number"])+"/"
    
    print("model_dir:", model_dir)
    sys.path.append(model_dir)
    print("sys path:", sys.path)
    
    from src.stylegan import StyleGan2, Dataset
    from src.dataloader import TiggeMRMSDataset
#     from run_src.utils import *
    
    print("Args loaded")
    # set seed
    torch.manual_seed(args.seed)

    ## Load Data and set data params
    
    print("Loading data ... ")
#     ds_valid = Dataset('/home/jupyter/data/stylegan2', 64)
    ds_train = pickle.load(open(args.data_hparams['train_dataset_path'], "rb"))
    ds_valid = pickle.load(open(args.data_hparams['valid_dataset_path'], "rb"))

#     import torchvision.datasets as datasets
#     import torchvision.transforms as transforms
#     transforms = transforms.Compose(
#     [
#         transforms.Resize(128),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5 for _ in range(1)], [0.5 for _ in range(1)]
#         ),
#     ]
#     )
#     ds_valid = datasets.MNIST(root="dataset/", transform=transforms, download=True)

    
    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.compute_weights(), len(ds_train))
    sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas = args.train_hparams['gpus'], rank = 0)


    sampler_valid = torch.utils.data.WeightedRandomSampler(ds_valid.compute_weights(), len(ds_valid))
    sampler_valid = DistributedSamplerWrapper(sampler_valid, num_replicas = args.train_hparams['gpus'], rank = 0)
    
    batch_size = args.train_hparams['batch_size']//args.train_hparams['gpus']
    
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=16, drop_last = True)

    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=16, drop_last = True)
    
#     dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, num_workers=16)

    
    print("Data loading complete")
    ## Load Model
    if input_args.ckpt_path:
        model = StyleGan2().load_from_checkpoint(input_args.ckpt_path)
    else:
        model = StyleGan2()

    ## Define trainer and logging

    save_dir = args.save_hparams['save_dir']

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir+args.save_hparams['run_name']+str(args.save_hparams['run_number']) + '/')
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = '../logs/',
                                             name = args.save_hparams['run_name'], 
                                             version = args.save_hparams['run_number'])

    
    if input_args.ckpt_path:
        trainer = pl.Trainer(accelerator='ddp', 
                         precision=16, gpus = args.train_hparams['gpus'], 
                         max_epochs = args.train_hparams['epochs'], 
                         callbacks=[checkpoint_callback], 
                         replace_sampler_ddp = False, 
                         check_val_every_n_epoch=20, 
                         logger = tb_logger, 
                         resume_from_checkpoint = input_args.ckpt_path
                        )
    else:
        trainer = pl.Trainer(
                            accelerator='ddp', 
                         precision=16, 
                         gpus = args.train_hparams['gpus'], 
                         max_epochs = args.train_hparams['epochs'], 
                         callbacks=[checkpoint_callback], 
                         replace_sampler_ddp = False, 
                         check_val_every_n_epoch=20, 
                         logger = tb_logger, 
#                          gradient_clip_val=1.0
#                          auto_select_gpus=True
                        )
                         
    print("Training model...")
    
    # Train
    trainer.fit(model, dl_train)    
    

if __name__ == '__main__':

    train(input_args)
    