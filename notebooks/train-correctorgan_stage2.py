import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.profiler import AdvancedProfiler

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from src.models import GANs, gens, discs
# # from ilan_src.models import *
# from src.dataloader import *
# from src.utils import *

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

    #save run config
    json.dump(vars(args), open(args.save_hparams["save_dir"]+args.save_hparams["run_name"]+str(args.save_hparams["run_number"])+"/experiment_config.json", 'w'))

    from run_src.models import GANs, gens, discs
    from run_src.dataloader import TiggeMRMSPatchLoadDataset
    from run_src.callbacks import StopIfNan

    
    print("Args loaded")
    # set seed
    torch.manual_seed(args.seed)
          
    args.gan_hparams['generator'] = gens[args.gan_hparams['generator']](input_channels = args.gan_hparams['input_channels'])

    ## Load Data and set data params
    
    print("Loading data ... ")
    ds_train = TiggeMRMSPatchLoadDataset(args.data_hparams['train_dataset_path'], samples_vars=args.data_hparams['samples_vars'])
    ds_valid = TiggeMRMSPatchLoadDataset(args.data_hparams['valid_dataset_path'], samples_vars=args.data_hparams['samples_vars'])
    
    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.weights, len(ds_train))
    sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas = len(args.train_hparams['gpus']) if type(args.train_hparams['gpus'])==list else  args.train_hparams['gpus'], rank = 0)
    sampler_valid = torch.utils.data.SequentialSampler(ds_valid)
    sampler_valid = DistributedSamplerWrapper(sampler_valid, num_replicas = len(args.train_hparams['gpus']) if type(args.train_hparams['gpus'])==list else  args.train_hparams['gpus'], rank = 0)
    
    
    if type(args.train_hparams['gpus']) == list:
        batch_size = args.train_hparams['batch_size']//len(args.train_hparams['gpus'])
    else:
        batch_size = args.train_hparams['batch_size']//args.train_hparams['gpus']
    
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=4, pin_memory=True)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=4, pin_memory=True)
        
    print("Data loading complete")
    # Load Model
    if input_args.ckpt_path:
        model = GANs[args.gan].load_from_checkpoint(input_args.ckpt_path)
        model.loss_hparams = args.gan_hparams['loss_hparams']
        model.opt_hparams = args.gan_hparams['opt_hparams']
        model.noise_shape = args.gan_hparams['noise_shape']
    else:
        raise NotImplementedError

    ## Define trainer and logging

    save_dir = args.save_hparams['save_dir']

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir+args.save_hparams['run_name']+str(args.save_hparams['run_number']) + '/', 
                                                      every_n_train_steps = 1000)
    
    
    nanloss_stopping = StopIfNan(monitor = ['generator_loss'], every_n_train_steps=100, model_save_every_n_train_steps=1000)
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = '../logs/',
                                             name = args.save_hparams['run_name'], 
                                             version = args.save_hparams['run_number'])

#     profiler = AdvancedProfiler(dirpath='./', filename='advanced_profile')
    
    if input_args.ckpt_path:
        trainer = pl.Trainer(accelerator='ddp', 
                         precision=16, gpus = args.train_hparams['gpus'], 
                         max_epochs = args.train_hparams['epochs'], 
                         callbacks=[checkpoint_callback, nanloss_stopping], 
                         replace_sampler_ddp = False, 
                         check_val_every_n_epoch=1, 
                         logger = tb_logger, 
                         resume_from_checkpoint = input_args.ckpt_path
                        )
    else:
        raise NotImplementedError
#     trainer = pl.Trainer(accelerator='ddp', 
#                          precision=16, gpus = args.train_hparams['gpus'], 
#                          max_epochs = args.train_hparams['epochs'], 
#                          callbacks=[checkpoint_callback, nanloss_stopping], 
#                          replace_sampler_ddp = False, 
#                          check_val_every_n_epoch=1, 
#                          logger = tb_logger, 
# #                          auto_select_gpus=True
# #                         accumulate_grad_batches=5, 
# #                         profiler="simple", 
# #                          max_steps=200
#                         )
                         
    print("Training model...")
    
    # Train
    trainer.fit(model, dl_train, dl_valid)    
    

if __name__ == '__main__':

    train(input_args)
    