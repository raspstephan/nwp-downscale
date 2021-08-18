import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchvision

import torch
import torch.nn as nn
from .utils import tqdm, device
from .layers import *
from .dataloader import log_retrans

import xarray as xr
import xskillscore as xs
from dask.diagnostics import ProgressBar
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt



######################################
# Stephan's original models
######################################


# options: 
# G: spectral_norm, noise, relu-out
# D: spectral_norm, batchnorm, sigmoid, conditional

class StephanGenerator(nn.Module):
    """Generator with noise vector and spectral normalization """
    def __init__(self, nres, nf_in, nf, activation_out=None, use_noise=True, spectral_norm=True,
                 nout=1, upsample_method='bilinear', halve_filters_up=True, 
                 batch_norm=False):
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
            ResidualBlock(nf, nf, spectral_norm=spectral_norm, batch_norm=batch_norm) for _ in range(nres)
        ])

        # Resblocks with upscaling (hardcoded for factor 8)
        self.upblocks = nn.Sequential(*[
            UpsampleBlock(
                nf//2**i if halve_filters_up else nf, 
                nf//2**(i+1)if halve_filters_up else nf, 
                spectral_norm=spectral_norm, 
                method=upsample_method,
                batch_norm=batch_norm
                ) 
                for i in range(3)
        ])

        # Final convolubtion
        self.conv_out = make_conv2d(
            nf//2**3 if halve_filters_up else nf, nout, kernel_size=3, padding=1, 
            spectral_norm=spectral_norm, padding_mode='reflect'
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


class StephanDiscriminator(nn.Module): 
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

class StephanDiscriminator2(nn.Module): 
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


######################################
# baseline gen and disc for WGAN GP
######################################

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size*img_size)
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img+1, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, labels, x):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):
    def __init__(self, noise_shape, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(noise_shape[0] + embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding  = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim = 1)
        x = self.net(x)
#         print(x.shape)
        return x

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

#####################################################
####################################################

class DSDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(DSDiscriminator, self).__init__()
        
        self.img_size = img_size
        self.embed = nn.Sequential(
                        # Input: N x channels_noise x 16 x 16
                        self._expand_block(channels_img, features_d * 16, 1, 1, 0),  # img: 16x16
                        self._expand_block(features_d * 16, features_d * 8, 4, 2, 1),  # img: 32x32
                        self._expand_block(features_d * 8, features_d * 4, 4, 2, 1),  # img: 64x64
                        nn.ConvTranspose2d(
                            features_d * 4, channels_img, kernel_size=4, stride=2, padding=1
                        ),
                        # Output: N x channels_img x 128 x 128
                        nn.Sigmoid(),
                    )
        self.disc = nn.Sequential(
            # input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img*2, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 6, 4, 2, 1),
            self._block(features_d * 6, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def _expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


class DSGenerator(nn.Module):
    def __init__(self, noise_shape, channels_img, features_g, num_classes, img_size, embed_size):
        super(DSGenerator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Conv2d(in_channels=1, out_channels=embed_size, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            # Input: N x channels_noise x 16 x 16
            self._block(noise_shape[0] + embed_size, features_g * 16, 1, 1, 0),  # img: 16x16
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 32x32
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 64x64
            nn.ConvTranspose2d(
                features_g * 4, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 128 x 128
            nn.Sigmoid(),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding  = self.embed(labels)
        x = torch.cat([x, embedding], dim = 1)
        x = self.net(x)
#         print(x.shape)
        return x

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                
                

class DSDiscriminatorSmoothed(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(DSDiscriminatorSmoothed, self).__init__()
        
        self.img_size = img_size
        self.embed = nn.Sequential(
                        # Input: N x channels_noise x 16 x 16
                        self._expand_block(channels_img, features_d * 16, kernel_size = 3, stride=1, padding=1, scale_factor=1),  # img: 16x16
                        self._expand_block(features_d * 16, features_d * 8, kernel_size = 3, stride=1, padding=1, scale_factor=2),  # img: 32x32
                        self._expand_block(features_d * 8, features_d * 4, kernel_size = 3, stride=1, padding=1, scale_factor=2),  # img: 64x64
                        nn.UpsamplingBilinear2d(scale_factor=2), 
                        nn.Conv2d(
                            features_d * 4, channels_img, kernel_size=3, stride=1, padding=1
                        ),
                        # Output: N x channels_img x 128 x 128
                        nn.Sigmoid(),
                    )
        self.disc = nn.Sequential(
            # input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img*2, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 6, 4, 2, 1),
            self._block(features_d * 6, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def _expand_block(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor):
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding = self.embed(labels)
#         print(embedding.shape)
#         embedding = embedding.view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

class DSGeneratorSmoothed(nn.Module):
    def __init__(self, noise_shape, channels_img, features_g, num_classes, img_size, embed_size):
        super(DSGeneratorSmoothed, self).__init__()
        self.img_size = img_size
        self.embed = nn.Conv2d(in_channels=1, out_channels=embed_size, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            # Input: N x channels_noise x 16 x 16
            self._block(noise_shape[0] + embed_size, features_g * 16, kernel_size=3, stride=1, padding=1, scale_factor=1),  # img: 16x16
            self._block(features_g * 16, features_g * 8, kernel_size=3, stride=1, padding=1, scale_factor=2),  # img: 32x32
            self._block(features_g * 8, features_g * 4, kernel_size=3, stride=1, padding=1, scale_factor=2),  # img: 64x64
            nn.UpsamplingBilinear2d(scale_factor=2), 
            nn.Conv2d(
                features_g * 4, channels_img, kernel_size=3, stride=1, padding=1
            ),
            # Output: N x channels_img x 128 x 128
            nn.Sigmoid(),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor):
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor), 
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding  = self.embed(labels)
        x = torch.cat([x, embedding], dim = 1)
        x = self.net(x)
#         print(x.shape)
        return x

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                
###################################################################
###################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=256, planes=256, stride=1, nonlin = 'relu'):
        super(BasicBlock, self).__init__()
        self.nonlin = nonlin
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.nonlin = nonlin

    def forward(self, x):
        if self.nonlin == 'leaky_relu':
            out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.02)
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.leaky_relu(out, negative_slope=0.02)
            return out
        elif self.nonlin == 'relu':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out
        else: raise NotImplementedError
            
class ConvBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size = 3, norm=None, stride=1, activation='leaky_relu', padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding,
                stride=stride)
        self.norm = norm
        self.activation = activation
        
    def forward(self, x):
        
        if self.norm=="batch":
            x = nn.BatchNorm2d(in_channels)(x)
        if self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.2)
        elif self.activation == 'relu':
            x = F.relu(x)
        x = self.conv(x)
        return x

        return block 
    
class LeinResBlock(nn.Module):

    def __init__(self, in_planes=256, planes=256, stride=1, nonlin = 'relu', norm =None):
        super(LeinResBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.nonlin = nonlin
        self.norm = norm
        
        shortcut_modules = []
        if self.stride>1:
            shortcut_modules.append(nn.AvgPool2d(self.stride))
        if (self.planes != self.in_planes):
                shortcut_modules.append(ConvBlock(self.in_planes, self.planes, 1, stride=1, 
                    activation=False, padding=0))
        
        self.shortcut = nn.Sequential(*shortcut_modules)   
        self.convblock1 = ConvBlock(self.in_planes, self.planes, 3, stride=self.stride,
            norm=self.norm,
            activation=self.nonlin)
        self.convblock2 = ConvBlock(self.planes, self.planes, 3, stride=1,
            norm=self.norm,
            activation=self.nonlin)
        
    def forward(self, x):
        x_in = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x_in = self.shortcut(x_in)
        x = x + x_in
        return x

    
            

            
class DeconvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=256, planes=256, stride=2, nonlin = 'relu'):
        super(DeconvBlock, self).__init__()
        self.nonlin = nonlin
        self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.nonlin = nonlin


    def forward(self, x):
        if self.nonlin == 'leaky_relu':
            out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.02)
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.leaky_relu(out, negative_slope=0.02)
            return out
        elif self.nonlin == 'relu':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out
        else: raise NotImplementedError
            

class UpSample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(UpSample, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
#         print(x.shape)
        return x
    
    
class LeinGen(nn.Module):
    def __init__(self, input_channels=1):
        super(LeinGen, self).__init__()
        self.embed = nn.Conv2d(input_channels,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'), 
                                      LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'), 
                            #         self.b3 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
#         print(x.shape)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
                                     
class LeinDisc(nn.Module):
    def __init__(self, input_channels=1, nonlin = 'leaky_relu'):
        super(LeinDisc, self).__init__()
        hr_block = []
        lr_block = []
        lr_inplanes = input_channels
        hr_inplanes = 1
        for planes in [64, 128, 256]:
            hr_block.append(LeinResBlock(in_planes = hr_inplanes, planes=planes, stride=2, nonlin = nonlin))
            lr_block.append(LeinResBlock(in_planes = lr_inplanes, planes=planes, stride=1, nonlin = nonlin))
            lr_inplanes=planes
            hr_inplanes=planes
        self.hr_block1 = nn.Sequential(*hr_block)
        self.lr_block1 = nn.Sequential(*lr_block)
        self.hr_block2 = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(LeinResBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.reshape(hr, (hr.shape[0],-1)), torch.reshape(lr, (lr.shape[0], -1))), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    
#     def spectral_norm(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 m = nn.utils.parametrizations.spectral_norm(m)
                                     
                                     
class LeinGen2(nn.Module):
    def __init__(self):
        super(LeinGen, self).__init__()
        self.embed = nn.Conv2d(1,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(BasicBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'), 
                                      BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'), 
                            #         self.b3 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(DeconvBlock(in_planes=256, planes=128, stride=2, nonlin = 'leaky_relu'), 
                                     DeconvBlock(in_planes=128, planes=64, stride=2,  nonlin = 'leaky_relu'), 
                                     DeconvBlock(in_planes=64, planes=32, stride=2,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    
class LeinDisc2(nn.Module):
    def __init__(self, nonlin = 'leaky_relu'):
        super(LeinDisc, self).__init__()
        hr_block = []
        lr_block = []
        inplanes = 1
        for planes in [64, 128, 256]:
            hr_block.append(BasicBlock(in_planes = inplanes, planes=planes, stride=2, nonlin = nonlin))
            lr_block.append(BasicBlock(in_planes = inplanes, planes=planes, stride=1, nonlin = nonlin))
            inplanes=planes
        self.hr_block1 = nn.Sequential(*hr_block)
        self.lr_block1 = nn.Sequential(*lr_block)
        self.hr_block2 = nn.Sequential(BasicBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(BasicBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
#         print(lr.shape)
#         print(hr.shape)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
        
        
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        
        super(SelfAttention, self).__init__()
        self.f = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, 
                           kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, 
                           kernel_size=1, stride=1, padding=0)        
        self.h = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                           kernel_size=1, stride=1, padding=0)
        
        gamma = torch.tensor(0.0)
        self.gamma = nn.Parameter(gamma, requires_grad=True)
        
        self.flatten = nn.Flatten()
        
    
    def collapse_height_width(self, x):
        x_shape = x.shape
        return torch.reshape(x, (x_shape[0], -1, x.shape[1]))
    
    
    def forward(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        
        f_flat = self.collapse_height_width(f)
        g_flat = self.collapse_height_width(g)
        h_flat = self.collapse_height_width(h)

#         print(g_flat.shape)
#         print(f_flat.shape)
        s = g_flat @ torch.transpose(f_flat, 1, 2)

#         print(s.shape)
        b = F.softmax(s, dim = -1)
        
        o = b @ h_flat
        
#         print(o.shape)
#         print(x.shape)
        y = self.gamma * torch.reshape(o, x.shape) + x
        
        return y
        
        
class BroadLeinSAGen(nn.Module):
    def __init__(self, input_channels=1):
        super(BroadLeinSAGen, self).__init__()
        self.embed = nn.Conv2d(input_channels,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=2,  nonlin = 'relu'), 
                                     LeinResBlock(in_planes=256, planes=256, stride=2, nonlin = 'relu'), 
#                                      LeinResBlock(in_planes=256, planes=256, stride=2, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     SelfAttention(256),
                                     LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
#         print(x.shape)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
#         print(x.shape)
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
                                     
class BroadLeinDisc(nn.Module):
    def __init__(self, input_channels = 1, nonlin = 'leaky_relu'):
        super(BroadLeinDisc, self).__init__()
        self.hr_block1 = nn.Sequential(LeinResBlock(in_planes = 1, planes=64, stride=2, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=2, nonlin = nonlin),
                                       LeinResBlock(in_planes = 128, planes=256, stride=2, nonlin = nonlin))
        
        self.lr_block1 = nn.Sequential(LeinResBlock(in_planes = input_channels, planes=64, stride=2, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=2, nonlin = nonlin),
                                       LeinResBlock(in_planes = 128, planes=256, stride=1, nonlin = nonlin))
        
        self.hr_block2 = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(LeinResBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
        
class BroadLeinGen(nn.Module):
    def __init__(self, input_channels=1):
        super(BroadLeinGen, self).__init__()
        self.embed = nn.Conv2d(input_channels,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(LeinResBlock(in_planes=255, planes=255, stride=2,  nonlin = 'leaky_relu'), 
                                     LeinResBlock(in_planes=255, planes=255, stride=2, nonlin = 'leaky_relu'), 
                                     LeinResBlock(in_planes=255, planes=255, stride=1, nonlin = 'leaky_relu'), 
                                     LeinResBlock(in_planes=255, planes=255, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'relu'))
        
        self.final = nn.Sequential(nn.Conv2d(32,16, kernel_size=3, padding=1), nn.LeakyReLU(0.02),
                                   nn.Conv2d(16,1, kernel_size=3, padding=1))
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = self.process(x)
        x = torch.cat((x,noise), axis=1)
#         print(x.shape)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
#         print(x.shape)
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
                                     
class BroadLeinSADisc(nn.Module):
    def __init__(self, input_channels = 1, nonlin = 'leaky_relu'):
        super(BroadLeinSADisc, self).__init__()
        self.hr_block1 = nn.Sequential(LeinResBlock(in_planes = 1, planes=64, stride=2, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=2, nonlin = nonlin),
                                       SelfAttention(128),
                                       LeinResBlock(in_planes = 128, planes=256, stride=2, nonlin = nonlin))
        
        self.lr_block1 = nn.Sequential(LeinResBlock(in_planes = input_channels, planes=64, stride=2, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=2, nonlin = nonlin),
                                       SelfAttention(128),
                                       LeinResBlock(in_planes = 128, planes=256, stride=1, nonlin = nonlin))
        
        self.hr_block2 = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(LeinResBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    

class LeinSAGen(nn.Module):
    def __init__(self, input_channels = 1):
        super(LeinSAGen, self).__init__()
        self.embed = nn.Conv2d(input_channels,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'), 
                                     LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'), 
#                                      LeinResBlock(in_planes=256, planes=256, stride=2, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     SelfAttention(256),
                                     LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
#         print(x.shape)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
#         print(x.shape)
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
                                     
class LeinSADisc(nn.Module):
    def __init__(self, input_channels = 1, nonlin = 'leaky_relu'):
        super(LeinSADisc, self).__init__()
        self.hr_block1 = nn.Sequential(LeinResBlock(in_planes = 1, planes=64, stride=2, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=2, nonlin = nonlin),
                                       SelfAttention(128),
                                       LeinResBlock(in_planes = 128, planes=256, stride=2, nonlin = nonlin))
        
        self.lr_block1 = nn.Sequential(LeinResBlock(in_planes = input_channels, planes=64, stride=1, nonlin = nonlin), 
                                       LeinResBlock(in_planes = 64, planes=128, stride=1, nonlin = nonlin),
                                       SelfAttention(128),
                                       LeinResBlock(in_planes = 128, planes=256, stride=1, nonlin = nonlin))
        
        self.hr_block2 = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(LeinResBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    
    

# class BroadCorrectionEmbeddingGen(nn.Module):
#     def __init__(self, input_channels=1):
#         super(BroadCorrectionEmbeddingGen, self).__init__()
#         self.embed = nn.Conv2d(input_channels,255, kernel_size=3, padding=1)
#         self.process = nn.Sequential(LeinResBlock(in_planes=255, planes=255, stride=2,  nonlin = 'relu'), 
#                                      LeinResBlock(in_planes=255, planes=255, stride=2, nonlin = 'relu'), 
#                                      LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'),
#                                      LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu')
#                                         )
#         self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'leaky_relu'),
#                                      UpSample(2, 'bilinear'),
#                                      LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'leaky_relu'),
#                                      UpSample(2, 'bilinear'),
#                                      LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'leaky_relu'),
#                                      UpSample(2, 'bilinear'),
#                                      LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'leaky_relu'))
        
#         self.final = nn.Sequential(nn.Conv2d(32,16, kernel_size=3, padding=1), nn.LeakyReLU(0.02), nn.Conv2d(16,1, kernel_size=3, padding=1))
         
#     def forward(self, x, noise):
#         x = F.relu(self.embed(x))
#         y = self.process(x)
#         x = torch.cat((x,noise), axis=1)
# #         print(x.shape)
#         x = self.upscale(x)
#         x = torch.sigmoid(self.final(x))
# #         print(x.shape)
#         return x
    
#     def initialize_weights(self):
#         # Initializes weights according to the DCGAN paper
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
# #                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#                 nn.init.kaiming_normal_(m.weight.data)
    
#########################################################
######## GAN MODELS ###################
##########################################################

class WGANGP(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, channels_img, features_g, num_classes, img_size, embed_size, features_d, 
                      b1 = 0.0, b2 = 0.9, lr = 1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1): # fill in
        super().__init__()
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.disc_freq, self.gen_freq = 5, 1
        self.noise_shape = noise_shape
        self.lambda_gp = lambda_gp
        self.gen = generator(noise_shape, channels_img, features_g, num_classes, img_size, embed_size)
        self.disc = discriminator(channels_img, features_d, num_classes, img_size)
        self.num_to_visualise = 12
        self.num_classes = num_classes
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        self.save_hyperparameters()
        
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%100==0:
            with torch.no_grad():
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
#                 print(sample_imgs.shape)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                
        
#         # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            fake = self.gen(condition, noise)
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake).reshape(-1)
            gp = self.gradient_penalty(condition, real, fake)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.lambda_gp*gp
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc

        
#         #train generator
        elif optimizer_idx ==1:
            noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            fake = self.gen(condition, noise)
            gen_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        disc_opt = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]

        
    
class LeinGANGP(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, channels_img, img_size, embed_size, 
                      b1 = 0.0, b2 = 0.9, lr = 1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1, disc_spectral_norm = False): # fill in
        super().__init__()
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.disc_freq, self.gen_freq = 5, 1
        self.noise_shape = noise_shape
        self.lambda_gp = lambda_gp
        self.gen = generator(channels_img)
        self.disc = discriminator(channels_img)
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        if disc_spectral_norm:   # TODO: Fix! currently does not work
            self.disc.apply(self.add_sn)
        self.save_hyperparameters()
        
    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(m)
        else:
            m
        
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%100==0:
            with torch.no_grad():
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
#                 print(sample_imgs.shape)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                
        
#         # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            fake = self.gen(condition, noise)
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake).reshape(-1)
            gp = self.gradient_penalty(condition, real, fake)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.lambda_gp*gp
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc

        
#         #train generator
        elif optimizer_idx ==1:
            noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            fake = self.gen(condition, noise)
            gen_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        disc_opt = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
#         return gen_opt, disc_opt




class BaseGAN(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, input_channels = 1,
                      b1 = 0.0, b2 = 0.9, disc_lr = 1e-4, gen_lr=1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1, 
                      gen_freq = 1, disc_freq=5, disc_spectral_norm = False, gen_spectral_norm = False, loss_type = "wasserstein", 
                        val_hparams = {'val_nens':10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0}): # fill in
        super().__init__()
        self.disc_lr, self.gen_lr, self.b1, self.b2 = disc_lr, gen_lr,  b1, b2
        self.disc_freq, self.gen_freq = disc_freq, gen_freq
        self.noise_shape = noise_shape
        self.lambda_gp = lambda_gp
        self.gen = generator(input_channels=input_channels)
        self.disc = discriminator(input_channels=input_channels)
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        self.loss_type = loss_type
        self.input_channels = input_channels
        self.val_hparams = val_hparams
        self.upsample_input = nn.Upsample(scale_factor=8)
        
        if disc_spectral_norm:  
            self.disc.apply(self.add_sn)
        if gen_spectral_norm:   
            self.gen.apply(self.add_sn)
        
        self.save_hyperparameters()
        
    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(m)
        else:
            m

                
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def loss_disc(self, disc_real, disc_fake):
        if self.loss_type == "wasserstein":
            return -(torch.mean(disc_real) - torch.mean(disc_fake))
        else:
            raise NotImplementedError
    
    def loss_gen(self, disc_fake):
        if self.loss_type == "wasserstein":
            return -torch.mean(disc_fake)
        else:
            raise NotImplementedError

    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%500==0:
                self.gen.eval()
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
#                 print(sample_imgs.shape)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                if self.input_channels>1:
                    input_forcasts = self.upsample_input(condition)
#                     print(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1).shape)
                    grid = torchvision.utils.make_grid(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1), nrow=self.input_channels)
                else:
                    grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                self.gen.train()
                
#         # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            fake = self.gen(condition, noise)
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake).reshape(-1)
            
            loss_disc = self.loss_disc(disc_real, disc_fake)
            
            if self.lambda_gp:
                gp = self.gradient_penalty(condition, real, fake)
                loss_disc = loss_disc + self.lambda_gp*gp
            
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc
        
#         #train generator
        elif optimizer_idx ==1:
            noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            fake = self.gen(condition, noise)
            disc_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = self.loss_gen(disc_fake)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=self.gen_lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        disc_opt = optim.Adam(self.disc.parameters(), lr=self.disc_lr, betas=(self.b1, self.b2))
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
#         return gen_opt, disc_opt

    def validation_step(self, batch, batch_idx):
        
        x, y = batch[self.cond_idx], batch[self.real_idx]

        preds = []
        for i in range(self.val_hparams['val_nens']):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=self.device)
            pred = self.gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.detach().to('cpu').numpy().squeeze(1)
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']

        preds = preds * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']
    
        if self.val_hparams['tp_log']:
            truth = log_retrans(truth, self.val_hparams['tp_log'])
            preds = log_retrans(preds, self.val_hparams['tp_log'])
            
        crps = []    
        rmse = []
        for sample in range(x.shape[0]):
            sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
            sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
            crps.append(sample_crps)
            rmse.append(sample_rmse)
            
        crps = torch.tensor(np.mean(crps), device=self.device)
        rmse = torch.tensor(np.mean(rmse), device=self.device)
        self.log('val_crps', crps, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        
        return crps

    
# class BaseGAN2(LightningModule):
#     def __init__(self, generator, discriminator, noise_shape, input_channels = 1,
#                       b1 = 0.0, b2 = 0.9, disc_lr = 1e-4, gen_lr=1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1, 
#                       gen_freq = 1, disc_freq=5, disc_spectral_norm = False, gen_spectral_norm = False, zero_noise = False,
#                       loss_hparams = {'disc_loss':"wasserstein", 'gen_loss':"wasserstein"}, 
#                       val_hparams = {'val_nens':10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0}): # fill in
#         super().__init__()
#         self.disc_lr, self.gen_lr, self.b1, self.b2 = disc_lr, gen_lr,  b1, b2
#         self.disc_freq, self.gen_freq = disc_freq, gen_freq
#         self.noise_shape = noise_shape
#         self.lambda_gp = lambda_gp
#         self.gen = generator(input_channels=input_channels)
#         self.disc = discriminator(input_channels=input_channels)
#         self.real_idx = real_idx
#         self.cond_idx = cond_idx
#         self.loss_hparams = loss_hparams
#         self.input_channels = input_channels
#         self.val_hparams = val_hparams
#         self.upsample_input = nn.Upsample(scale_factor=8)
#         self.zero_noise = zero_noise
        
#         if disc_spectral_norm:  
#             self.disc.apply(self.add_sn)
#         if gen_spectral_norm:   
#             self.gen.apply(self.add_sn)
        
#         self.save_hyperparameters()
        
#     def add_sn(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
#             nn.utils.spectral_norm(m)
#         else:
#             m

                
#     def forward(self, condition, noise):
#         return self.gen(condition, noise)
    
#     def gradient_penalty(self, condition, real, fake):
#         BATCH_SIZE, C, H, W = real.shape
#         epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
#         interpolated_images = real*epsilon + fake*(1-epsilon)
#         interpolated_images.requires_grad = True
#         mixed_scores = self.disc(condition, interpolated_images)
        
#         gradient = torch.autograd.grad(
#                     inputs=interpolated_images,
#                     outputs=mixed_scores, 
#                     grad_outputs = torch.ones_like(mixed_scores), 
#                     create_graph=True, 
#                     retain_graph = True)[0]

#         gradient = gradient.view(gradient.shape[0], -1)
#         gradient_norm = gradient.norm(2, dim=1)
#         gradient_penalty = torch.mean((gradient_norm - 1)**2)
#         return gradient_penalty
    
#     def loss_disc(self, disc_real, disc_fake):
#         if self.loss_hparams['disc_loss'] == "wasserstein":
#             return -(torch.mean(disc_real) - torch.mean(disc_fake))
#         elif self.loss_hparams['disc_loss'] == "hinge":
#             return torch.mean(F.relu(1-disc_real)) + torch.mean(F.relu(1+disc_fake))
#         else:
#             raise NotImplementedError
    
#     def loss_gen(self, fake, disc_fake, real):
#         if self.loss_hparams['gen_loss'] == "wasserstein":
#             return -torch.mean(disc_fake)
#         elif self.loss_hparams['gen_loss'] == "ens_mean_L1_weighted":
#             assert len(self.noise_shape)==4
#             l = -torch.mean(disc_fake)
# #             print('loss in loss gen', l)
#             lambda_l1_reg = self.loss_hparams['lambda_l1_reg']
#             mean_fake = torch.mean(fake, dim=0)
# #             print("mean fake")
# #             print(mean_fake)
#             diff = mean_fake - real
# #             print("diff", diff)
#             def weight_diff(y):
#                 return torch.clamp(y+1, min=24)
#             clipped = weight_diff(real)
# #             print("clipped", clipped)
#             weighted_diff = diff * clipped
# #             print("weighted_diff", weighted_diff)
# #             print("weighted_diff shape", weighted_diff.shape)
#             l += lambda_l1_reg*(1/real.numel()) * torch.linalg.norm(weighted_diff.reshape(-1), 1)
# #             print('loss in loss gen', l)
#             return l
#         else:
#             raise NotImplementedError

    
#     def training_step(self, batch, batch_idx, optimizer_idx):

#         condition, real = batch[self.cond_idx], batch[self.real_idx]

#         if self.global_step%500==0:
#                 self.gen.eval()
#                 noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
#         #         # log sampled images
#                 sample_imgs = self.gen(condition, noise)
#                 sample_imgs = torch.cat([real, sample_imgs], dim = 0)
# #                 print(sample_imgs.shape)
#                 grid = torchvision.utils.make_grid(sample_imgs)
#                 self.logger.experiment.add_image('generated_images', grid, self.global_step)
#                 if self.input_channels>1:
#                     input_forcasts = self.upsample_input(condition)
# #                     print(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1).shape)
#                     grid = torchvision.utils.make_grid(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1), nrow=self.input_channels)
#                 else:
#                     grid = torchvision.utils.make_grid(condition)
#                 self.logger.experiment.add_image('input_images', grid, self.global_step)
#                 self.gen.train()
        
# #         # train discriminator
#         if optimizer_idx == 0:
#             if self.zero_noise:
#                 noise = torch.zeros(real.shape[0], *self.noise_shape[-3:], device=self.device)
#             else:
#                 noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
#             disc_real = self.disc(condition, real).reshape(-1)
#             if len(noise.shape) == 5:
#                 fakes = []
#                 disc_fakes = []
#                 for i in range(noise.shape[1]):
#                     noise_sample = noise[:,i,:,:,:]
#                     fake = self.gen(condition, noise_sample)
#                     disc_fake = self.disc(condition, fake).reshape(-1)
#                     fakes.append(fake)
#                     disc_fakes.append(disc_fake)
#                 fakes = torch.stack(fakes, dim=0)
#                 disc_fakes = torch.stack(disc_fakes, dim = 0)
# #                 print("fakes.shape", fakes.shape)
# #                 print("disc_fakes.shape", disc_fakes.shape)
#                 loss_disc = self.loss_disc(disc_real, disc_fakes)
# #                 print("disc loss", loss_disc)
#             else:
#                 fake = self.gen(condition, noise)
#                 disc_fake = self.disc(condition, fake).reshape(-1)
#                 loss_disc = self.loss_disc(disc_real, disc_fake)
            
#             if self.lambda_gp:
#                 gp = self.gradient_penalty(condition, real, fake)
#                 loss_disc = loss_disc + self.lambda_gp*gp
            
#             self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
#             return loss_disc
        
# #         #train generator
#         elif optimizer_idx ==1:
# #             print(self.gen.training)
#             if self.zero_noise:
#                 noise = torch.zeros(real.shape[0], *self.noise_shape, device = self.device)
#             else:
#                 noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
#             if len(noise.shape) == 5:
#                 fakes = []
#                 disc_fakes = []
#                 for i in range(noise.shape[1]):
#                     noise_sample = noise[:,i,:,:,:]
#                     fake = self.gen(condition, noise_sample)
#                     disc_fake = self.disc(condition, fake).reshape(-1)
#                     fakes.append(fake)
#                     disc_fakes.append(disc_fake)
#                 fakes = torch.stack(fakes, dim=0)
#                 disc_fakes = torch.stack(disc_fakes, dim = 0)
# #                 print("fakes.shape", fakes.shape)
# #                 print("disc_fakes.shape", disc_fakes.shape)
#                 loss_gen = self.loss_gen(fakes, disc_fakes, real)
#             else:
#                 fake = self.gen(condition, noise)
#                 disc_fake = self.disc(condition, fake).reshape(-1)
#                 loss_gen = self.loss_gen(fake, disc_fake, real)
#             self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
#             return loss_gen 
        
        
#     def configure_optimizers(self):
#         gen_opt = optim.Adam(self.gen.parameters(), lr=self.gen_lr, betas=(self.b1, self.b2), weight_decay=1e-4)
#         disc_opt = optim.Adam(self.disc.parameters(), lr=self.disc_lr, betas=(self.b1, self.b2))
#         return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
# #         return gen_opt, disc_opt

#     def validation_step(self, batch, batch_idx):
        
#         x, y = batch[self.cond_idx], batch[self.real_idx]

#         preds = []
#         for i in range(self.val_hparams['val_nens']):
#             noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=self.device)
#             pred = self.gen(x, noise).detach().to('cpu').numpy().squeeze()
#             preds.append(pred)
#         preds = np.array(preds)
#         truth = y.detach().to('cpu').numpy().squeeze(1)
#         truth = xr.DataArray(
#                 truth,
#                 dims=['sample','lat', 'lon'],
#                 name='tp'
#             )
#         preds = xr.DataArray(
#                 preds,
#                 dims=['member', 'sample', 'lat', 'lon'],
#                 name='tp'
#             )

#         truth = truth * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']

#         preds = preds * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']
    
#         if self.val_hparams['tp_log']:
#             truth = log_retrans(truth, self.val_hparams['tp_log'])
#             preds = log_retrans(preds, self.val_hparams['tp_log'])
            
#         crps = []    
#         rmse = []
#         for sample in range(x.shape[0]):
#             sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
#             sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
#             crps.append(sample_crps)
#             rmse.append(sample_rmse)
            
#         crps = torch.tensor(np.mean(crps), device=self.device)
#         rmse = torch.tensor(np.mean(rmse), device=self.device)
#         self.log('val_crps', crps, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
#         self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        
#         return crps
    

class BaseGAN2(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, input_channels = 1,
                      cond_idx = 0, real_idx = 1, 
                      disc_spectral_norm = False, gen_spectral_norm = False, zero_noise = False,
                      opt_hparams = {'gen_optimiser':'adam', 'disc_optimiser':'adam', 'disc_lr' : 1e-4, 'gen_lr': 1e-4, 'gen_freq' : 1, 'disc_freq':5, 'b1':0.0, 'b2' : 0.9},
                      loss_hparams = {'disc_loss': "wasserstein", 'gen_loss':"wasserstein", 'lambda_gp': 10}, 
                      val_hparams = {'val_nens':10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0}): # fill in
        super().__init__()
        
        self.noise_shape = noise_shape
        self.gen = generator(input_channels=input_channels)
        self.disc = discriminator(input_channels=input_channels)
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        self.opt_hparams = opt_hparams
        self.loss_hparams = loss_hparams
        self.input_channels = input_channels
        self.val_hparams = val_hparams
        self.upsample_input = nn.Upsample(scale_factor=8)
        self.zero_noise = zero_noise
        
        if disc_spectral_norm:  
            self.disc.apply(self.add_sn)
        if gen_spectral_norm:   
            self.gen.apply(self.add_sn)
        
        self.save_hyperparameters()
        
    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(m)
        else:
            m

                
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def loss_disc(self, disc_real, disc_fake):
        if self.loss_hparams['disc_loss'] == "wasserstein":
            return -(torch.mean(disc_real) - torch.mean(disc_fake))
        elif self.loss_hparams['disc_loss'] == "hinge":
            return torch.mean(F.relu(1-disc_real)) + torch.mean(F.relu(1+disc_fake))
        else:
            raise NotImplementedError
    
    def loss_gen(self, fake, disc_fake, real):
        if self.loss_hparams['gen_loss'] == "wasserstein":
            return -torch.mean(disc_fake)
        elif self.loss_hparams['gen_loss'] == "ens_mean_L1_weighted":
            assert len(self.noise_shape)==4
            l = -torch.mean(disc_fake)
#             print('loss in loss gen', l)
            lambda_l1_reg = self.loss_hparams['lambda_l1_reg']
            mean_fake = torch.mean(fake, dim=0)
#             print("mean fake")
#             print(mean_fake)
            diff = mean_fake - real
#             print("diff", diff)
            def weight_diff(y):
                return torch.clamp(y+1, min=24)
            clipped = weight_diff(real)
#             print("clipped", clipped)
            weighted_diff = diff * clipped
#             print("weighted_diff", weighted_diff)
#             print("weighted_diff shape", weighted_diff.shape)
            l += lambda_l1_reg*(1/real.numel()) * torch.linalg.norm(weighted_diff.reshape(-1), 1)
#             print('loss in loss gen', l)
            return l
        else:
            raise NotImplementedError

    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%500==0:
                self.gen.eval()
                noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
#                 print(sample_imgs.shape)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                if self.input_channels>1:
                    input_forcasts = self.upsample_input(condition)
#                     print(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1).shape)
                    grid = torchvision.utils.make_grid(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1), nrow=self.input_channels)
                else:
                    grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                self.gen.train()
        
#         # train discriminator
        if optimizer_idx == 0:
            if self.zero_noise:
                noise = torch.zeros(real.shape[0], *self.noise_shape[-3:], device=self.device)
            else:
                noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            disc_real = self.disc(condition, real).reshape(-1)
            if len(noise.shape) == 5:
                fakes = []
                disc_fakes = []
                for i in range(noise.shape[1]):
                    noise_sample = noise[:,i,:,:,:]
                    fake = self.gen(condition, noise_sample)
                    disc_fake = self.disc(condition, fake).reshape(-1)
                    fakes.append(fake)
                    disc_fakes.append(disc_fake)
                fakes = torch.stack(fakes, dim=0)
                disc_fakes = torch.stack(disc_fakes, dim = 0)
#                 print("fakes.shape", fakes.shape)
#                 print("disc_fakes.shape", disc_fakes.shape)
                loss_disc = self.loss_disc(disc_real, disc_fakes)
#                 print("disc loss", loss_disc)
            else:
                fake = self.gen(condition, noise)
                disc_fake = self.disc(condition, fake).reshape(-1)
                loss_disc = self.loss_disc(disc_real, disc_fake)
            
            if 'lambda_gp' in self.loss_hparams:
                gp = self.gradient_penalty(condition, real, fake)
                loss_disc = loss_disc + self.loss_hparams['lambda_gp']*gp
            
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc
        
#         #train generator
        elif optimizer_idx ==1:
#             print(self.gen.training)
            if self.zero_noise:
                noise = torch.zeros(real.shape[0], *self.noise_shape, device = self.device)
            else:
                noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            if len(noise.shape) == 5:
                fakes = []
                disc_fakes = []
                for i in range(noise.shape[1]):
                    noise_sample = noise[:,i,:,:,:]
                    fake = self.gen(condition, noise_sample)
                    disc_fake = self.disc(condition, fake).reshape(-1)
                    fakes.append(fake)
                    disc_fakes.append(disc_fake)
                fakes = torch.stack(fakes, dim=0)
                disc_fakes = torch.stack(disc_fakes, dim = 0)
#                 print("fakes.shape", fakes.shape)
#                 print("disc_fakes.shape", disc_fakes.shape)
                loss_gen = self.loss_gen(fakes, disc_fakes, real)
            else:
                fake = self.gen(condition, noise)
                disc_fake = self.disc(condition, fake).reshape(-1)
                loss_gen = self.loss_gen(fake, disc_fake, real)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        if self.opt_hparams['gen_optimiser'] == 'adam':
            gen_opt = optim.Adam(self.gen.parameters(), lr=self.opt_hparams['gen_lr'], betas=(self.opt_hparams['b1'], self.opt_hparams['b2']), weight_decay=1e-4)
            
        elif self.opt_hparams['gen_optimiser'] == 'sgd':
            gen_opt = optim.SGD(self.gen.parameters(), lr=self.opt_hparams['gen_lr'], momentum = self.opt_hparams['gen_momentum'])
        else:
            raise NotImplementedError
        if self.opt_hparams['disc_optimiser'] == 'adam':
            disc_opt = optim.Adam(self.disc.parameters(), lr=self.opt_hparams['disc_lr'], betas=(self.opt_hparams['b1'], self.opt_hparams['b2']))
        elif self.opt_hparams['disc_optimiser'] == 'sgd':
            disc_opt = optim.SGD(self.disc.parameters(), lr=self.opt_hparams['disc_lr'], momentum = self.opt_hparams['disc_momentum'])
        else:
            raise NotImplementedError
        return [{"optimizer": disc_opt, "frequency": self.opt_hparams['disc_freq']}, {"optimizer": gen_opt, "frequency": self.opt_hparams['gen_freq']}]

    def validation_step(self, batch, batch_idx):
        
        x, y = batch[self.cond_idx], batch[self.real_idx]

        preds = []
        for i in range(self.val_hparams['val_nens']):
            noise = torch.randn(y.shape[0], *self.noise_shape[-3:], device=self.device)
            pred = self.gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.detach().to('cpu').numpy().squeeze(1)
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']

        preds = preds * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']
    
        if self.val_hparams['tp_log']:
            truth = log_retrans(truth, self.val_hparams['tp_log'])
            preds = log_retrans(preds, self.val_hparams['tp_log'])
            
        crps = []    
        rmse = []
        for sample in range(x.shape[0]):
            sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
            sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
            crps.append(sample_crps)
            rmse.append(sample_rmse)
            
        crps = torch.tensor(np.mean(crps), device=self.device)
        rmse = torch.tensor(np.mean(rmse), device=self.device)
        self.log('val_crps', crps, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        
        return crps
    
GANs = {
    'base':BaseGAN, 
    'wgan-gp':WGANGP, 
    'base2':BaseGAN2,
}

gens = {
    'leingen':LeinGen, 
    'broadleingen':BroadLeinGen, 
    'leinsagen':LeinSAGen, 
    'broadleinsagen':BroadLeinSAGen, 
}

discs = {
    'leindisc':LeinDisc,
    'broadleindisc':BroadLeinDisc, 
    'leinsadisc':LeinSADisc, 
    'broadleinsadisc': BroadLeinSADisc
}