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


# def plot_samples_per_input(data, gen, samples, device, save_dir):
#     fig, axs = plt.subplots(len(data), samples+2, figsize=(5*samples, len(data)*samples))
#     gen_images = np.zeros((len(data),samples,128,128))
#     for i, d in enumerate(data):
#         for s in range(samples):
#             cond = torch.tensor(d[0]).unsqueeze(0).to(device)
#             noise = torch.randn(cond.shape[0], 1, cond.shape[2], cond.shape[3]).to(device)
#             pred = gen(cond, noise).detach().cpu().numpy()
#             gen_images[i, s, :, :] = pred[0,0,:,:]
    
#     for i, d in enumerate(data):
#         cond = torch.tensor(d[0]).unsqueeze(0).to(device)
#         lr = cond[0,0,:,:].detach().cpu().numpy()
#         if lr.shape[0]==64:
#             lr = lr[24:40, 24:40]
#         hr = d[1][0,:,:]
#         mn = np.min([np.min(hr), np.min(pred), np.min(gen_images[i,:,:,:])])
#         mx = np.max([np.max(hr), np.max(pred), np.max(gen_images[i,:,:,:])])
#         im = axs[i,0].imshow(lr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
# #         plt.colorbar(im, ax=axs[i,0])
#         im = axs[i,1].imshow(hr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
# #         plt.colorbar(im, ax=axs[j,0], shrink=0.7)
#         for j in range(samples):
#             im = axs[i,j+2].imshow(gen_images[i,j,:,:], vmin=mn, vmax=mx, cmap='gist_ncar_r')
# #             plt.colorbar(im, ax=axs[j,i], shrink=0.7)
# #     plt.show()  
#     cols = ["Forecast", "Ground Truth"] + [f"Prediction {i+1}" for i in range(samples)]
#     for ax, col in zip(axs[0], cols):
#         ax.set_title(col)
#     plt.tight_layout()
#     plt.savefig(save_dir+'sample_predictions.png', dpi=400)
    
def plot_sr_of_stacked_input(data, gen, samples, device, save_dir):
    fig, axs = plt.subplots(len(data), 2*samples+1, figsize=(5*samples, len(data)*samples))
    gen_images = np.zeros((len(data),samples,128,128))
    lr_images = np.zeros((len(data),samples,16,16))
    true_images = np.zeros((len(data),128,128))
    for i, d in enumerate(data):
        true_images[i, :, :] = d[1]
        for s in range(samples):
#             print(d[0].shape)
            cond = torch.tensor(d[0])[s:s+1, :, :].unsqueeze(0).to(device)
#             print(cond.shape)
            noise = torch.zeros(cond.shape[0], 1, cond.shape[2], cond.shape[3]).to(device)
            pred = gen(cond, noise).detach().cpu().numpy()
            gen_images[i, s, :, :] = pred[0,0,:,:]
            lr_images[i, s, :, :] = cond.squeeze().detach().cpu().numpy()
                    
    for i, d in enumerate(data):
        mn = np.min([np.min(true_images[i,:,:]), np.min(lr_images[i,:,:,:]), np.min(gen_images[i,:,:,:])])
        mx = np.max([np.max(true_images[i,:,:]), np.max(lr_images[i,:,:,:]), np.max(gen_images[i,:,:,:])])
        im = axs[i,0].imshow(true_images[i,:,:], vmin=mn, vmax=mx, cmap='gist_ncar_r')
        for s in range(samples):    
            im = axs[i,2*s +1 ].imshow(lr_images[i,s,:,:], vmin=mn, vmax=mx, cmap='gist_ncar_r')
            im = axs[i,2*s +2].imshow(gen_images[i,s,:,:], vmin=mn, vmax=mx, cmap='gist_ncar_r')
            #             plt.colorbar(im, ax=axs[j,i], shrink=0.7)
#     plt.show()  
    def flatten(t):
        return [item for sublist in t for item in sublist]
    cols = ["Ground Truth"] + flatten([[f"Forecast{i+1}", f"Prediction {i+1}"] for i in range(samples)])
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(save_dir+'sample_predictions.png', dpi=400)
    
def visualise(input_args):
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

    #set seed
    torch.manual_seed(args.seed)

    ## Load Data and set data params
    ds_test = pickle.load(open(args.data_hparams["test_dataset_path"], "rb"))
    
    sample_indices = pickle.load(open("./sample_indices.pkl", 'rb'))
    ds_test = torch.utils.data.Subset(ds_test, sample_indices)
    
    print("Loading data ... ")

    gan = GANs[args.gan].load_from_checkpoint(args.model_path)
    
    gen = gan.gen
    gen = gen.to(device)
    gen.train(False);

    print("Data loading complete")
    
    plot_sr_of_stacked_input(ds_test, gen, 4, device, model_dir) 
    

if __name__ == '__main__':
    visualise(input_args)