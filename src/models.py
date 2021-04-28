import torch
import torch.nn as nn
from .utils import tqdm, device
from .layers import *
import numpy as np
import matplotlib.pyplot as plt




# options: 
# G: spectral_norm, noise, relu-out
# D: spectral_norm, batchnorm, sigmoid, conditional

class Generator(nn.Module):
    """Generator with noise vector and spectral normalization """
    def __init__(self, nres, nf_in, nf, activation_out=None, use_noise=True, spectral_norm=True,
                 nout=1, softmax_out=False, upsample_method='bilinear'):
        """ General Generator with different options to use. e.g noise, Spectral normalization (SN) """
        super().__init__()
        self.activation_out = activation_out
        self.use_noise = use_noise

        # First convolution
        self.conv_in = make_conv2d(
            nf_in, nf-1 if use_noise else nf, kernel_size=3, padding=1, spectral_norm=spectral_norm,
            padding_mode='reflect'
            )
        self.activation_in = nn.LeakyReLU(0.2)

        # Resblocks keeping shape
        self.resblocks = nn.Sequential(*[
            ResidualBlock(nf, nf, spectral_norm=spectral_norm) for _ in range(nres)
        ])

        # Resblocks with upscaling (hardcoded for factor 8 and halving nf)
        self.upblocks = nn.Sequential(*[
            UpsampleBlock(nf//2**i, nf//2**(i+1), spectral_norm=spectral_norm, method=upsample_method) for i in range(3)
        ])

        # Final convolubtion
        self.conv_out = make_conv2d(
            nf//2**3, nout, kernel_size=3, padding=1, spectral_norm=spectral_norm, padding_mode='reflect'
            )

        
    def forward(self, x):
        out = self.conv_in(x)
        out = self.activation_in(out)
        if self.use_noise: 
            bs, _, h, w = x.shape
            z = torch.normal(0, 1, size=(bs, 1, h, w), device=device, requires_grad=True)
            out = torch.cat([out, z], dim=1)
        out = self.resblocks(out)
        out = self.upblocks(out)
        out = self.conv_out(out)
        if self.activation_out == 'relu':
            out = nn.functional.relu(out)
        if self.activation_out == 'sigmoid':
            out = nn.functional.sigmoid(out)
        if self.activation_out == 'softmax':
            out = nn.functional.softmax(out, dim=1)
        return out


class Discriminator(nn.Module): 
    """ A first simple discriminator with binary output (sigmoid at final layer)"""
    def __init__(self, nf, ndown, nres, batch_norm=False, sigmoid=False, conditional=True, spectral_norm=True):
        """ General form of a Discriminator with different options to choose.
        
        batch_norm: If True, batch norm is applied in the Discriminator blocks. 
        sigmoid: whether to apply the sigmoid at the end. Set to False, e.g. for WGAN to get non-binary output.
        conditional: If True, conditional Disc. takes also low-res image as input in addition to high res image
        spectral_norm: If True, spectral normalization is applied.         
        """
        # Initialize object: 
        super().__init__()
        self.batch_norm = batch_norm
        assert self.batch_norm is False, 'Batch norm not implemented at the moment'
        self.sigmoid = sigmoid
        self.conditional = conditional
        
        if self.conditional: 
            self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
            
        # Down layers (Hard coded channels and number of layers)
        self.downblocks = []
        nf_in = 2 if self.upsample else 1; nf_out = nf//2**(ndown-1)
        for i in range(ndown):
            self.downblocks.append(
                ResidualBlock(nf_in, nf_out, stride=2, spectral_norm=spectral_norm)
            )
            nf_in = nf_out; nf_out *= 2
        self.downblocks = nn.Sequential(*self.downblocks)

        # Resblocks
        self.resblocks = nn.Sequential(*[
            ResidualBlock(nf, nf, spectral_norm) for _ in range(nres)
        ])

        # Dense layers

        linear1 = nn.Linear(nf, nf//2)
        linear2 = nn.Linear(nf//2, 1)
        if spectral_norm:
            linear1 = torch.nn.utils.spectral_norm(linear1)
            linear2 = torch.nn.utils.spectral_norm(linear2)
        self.final_layers = nn.Sequential(
            linear1,
            nn.LeakyReLU(0.2),
            linear2,
        )

    def forward(self, x):
        if self.conditional: # concatenate low and high res images by simply upsampling of low-res image.
            lr, hr = x
            x = torch.cat([hr, self.upsample(lr)], dim=1)            
        out = self.downblocks(x)
        out = self.resblocks(out)
        out = out.mean([2, 3])  # Global average pooling
        out = self.final_layers(out)
        if self.sigmoid:
            out = nn.functional.sigmoid(out)
        return out

class Discriminator2(nn.Module): 
    """ A first simple discriminator with binary output (sigmoid at final layer)"""
    def __init__(self, nf, nres, batch_norm=False, sigmoid=False, 
    spectral_norm=True):
        """ General form of a Discriminator with different options to choose.
        
        batch_norm: If True, batch norm is applied in the Discriminator blocks. 
        sigmoid: whether to apply the sigmoid at the end. Set to False, e.g. for WGAN to get non-binary output.
        conditional: If True, conditional Disc. takes also low-res image as input in addition to high res image
        spectral_norm: If True, spectral normalization is applied.         
        """
        # Initialize object: 
        super().__init__()
        self.batch_norm = batch_norm
        assert self.batch_norm is False, 'Batch norm not implemented at the moment'
        self.sigmoid = sigmoid
            
        # Down layers (Hard coded channels and number of layers)
        self.downblocksHR = []
        self.resblocksLR = []
        nf_in = 1; nf_out = nf//2**(3-1)
        for i in range(3):
            self.downblocksHR.append(
                ResidualBlock(nf_in, nf_out, stride=2, spectral_norm=spectral_norm)
            )
            self.resblocksLR.append(
                ResidualBlock(nf_in, nf_out, spectral_norm=spectral_norm)
            )
            nf_in = nf_out; nf_out *= 2
        self.downblocksHR = nn.Sequential(*self.downblocksHR)
        self.resblocksLR = nn.Sequential(*self.resblocksLR)        

        # Resblocks
        self.resblocks = nn.Sequential(
            ResidualBlock(nf*2, nf, spectral_norm),
            *[
                ResidualBlock(nf, nf, spectral_norm) for _ in range(nres-1)
        ])

        # Dense layers

        linear1 = nn.Linear(nf, nf//2)
        linear2 = nn.Linear(nf//2, 1)
        if spectral_norm:
            linear1 = torch.nn.utils.spectral_norm(linear1)
            linear2 = torch.nn.utils.spectral_norm(linear2)
        self.final_layers = nn.Sequential(
            linear1,
            nn.LeakyReLU(0.2),
            linear2,
        )

    def forward(self, x):
        lr, hr = x
        hr = self.downblocksHR(hr)
        lr = self.resblocksLR(lr)
        out = torch.cat([hr, lr], dim=1)
        out = self.resblocks(out)
        out = out.mean([2, 3])  # Global average pooling
        out = self.final_layers(out)
        if self.sigmoid:
            out = nn.functional.sigmoid(out)
        return out
