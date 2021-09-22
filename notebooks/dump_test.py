import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models import *
from src.dataloader import *
from src.utils import *

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pickle
import json

from pytictoc import TicToc

t = TicToc()

t.tic()

data_dir = '/home/jupyter/data/'

val_args = pickle.load(open('/home/jupyter/data/data_patches/valid/configs/dataset_args.pkl', 'rb'))

args = {'href_dir': data_dir + 'hrefv2/4km/total_precipitation/first5/*.nc',
    'tigge_dir':data_dir + f'tigge/32km/',
    'tigge_vars':['total_precipitation_ens10','total_column_water', '2m_temperature', 'convective_available_potential_energy', 'convective_inhibition'],
    'mrms_dir':data_dir + f'mrms/4km/RadarOnly_QPE_06H/',
    'rq_fn':data_dir + f'mrms/4km/RadarQuality.nc',
#     'const_fn':data_dir + 'tigge/32km/constants.nc',
#     'const_vars':['orog', 'lsm'],
    'data_period':('2020-01', '2020-12'),
    'first_days':5,
    'tp_log':0.01, 
    'scale':True,
    'ensemble_mode':'stack_by_variable',
    'pad_tigge':15,
    'pad_tigge_channel': True, 
    'idx_stride': 8, 
    'maxs': val_args['maxs'],
    'mins': val_args['mins']
    }

pickle.dump(args, open('/home/jupyter/data/data_patches/test/configs/dataset_args.pkl', 'wb'))

save_dir = '/home/jupyter/data/data_patches/'

ds_test = TiggeMRMSHREFDataset(**args)

t.toc('Loading dataset took')

t.tic()

starting_idx=0
save_images(ds_test, save_dir, 'test', starting_idx)

t.toc('Saving patches took')