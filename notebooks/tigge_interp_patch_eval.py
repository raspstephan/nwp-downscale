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


def evaluate():
    
    device = torch.device(f'cuda:0')
    print("device", device)
    
    from src.dataloader import TiggeMRMSDataset
#     from run_src.utils import *
    from src.evaluation import tigge_interp_patch_eval

    #set seed
    torch.manual_seed(0)

    print("Loading data ... ")
    ## Load Data and set data params
    ds_test = pickle.load(open("/home/jupyter/data/saved_datasets/testdataset_ensemble_tp_x10_stacked_log_trans_first_days_5.pkl", "rb"))  
    sampler_test = torch.utils.data.SequentialSampler(ds_test)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=128, sampler=sampler_test
    )

    
    print("Data loading complete")
#     print("ds test type", type(ds_test))
    ## Load Model
    metrics = tigge_interp_patch_eval(dl_test, ds_test.mins.tp.values, ds_test.maxs.tp.values, ds_test.tp_log, device)    
        
    print(metrics)

    pickle.dump(metrics, open("/home/jupyter/data/saved_models/interp_tigge_patch_eval_metrics.pkl", "wb"))

if __name__ == '__main__':
    evaluate()