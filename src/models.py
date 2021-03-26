import torch
import torch.nn as nn
from .utils import tqdm, device
import numpy as np
import matplotlib.pyplot as plt




# options: 
# G: spectral_norm, noise, relu-out
# D: spectral_norm, batchnorm, sigmoid, conditional



class ResidualBlock(nn.Module):
    def __init__(self, nf, spectral_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        if spectral_norm:  # use spectral normalization
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x
    
class UpsampleBlock(nn.Module):
    def __init__(self, nf, spectral_norm=False, method='PixelShuffle'):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf * 4 if method=='PixelShuffle' else nf, kernel_size=3, stride=1, padding=1)
        if method == 'PixelShuffle':
            self.upsample = nn.PixelShuffle(2)
        elif method == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            raise NotImplementedError
        self.activation = nn.LeakyReLU(0.2)
        if spectral_norm: 
            self.conv = nn.utils.spectral_norm(self.conv)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        out = self.activation(out)
        return out

class Generator(nn.Module):
    """Generator with noise vector and spectral normalization """
    def __init__(self, nres, nf_in, nf, relu_out=False, use_noise=True, spectral_norm=True,
                 nout=1, softmax_out=False, upsample_method='PixelShuffle'):
        """ General Generator with different options to use. e.g noise, Spectral normalization (SN) """
        super().__init__()
        self.relu_out = relu_out
        self.softmax_out = softmax_out
        self.use_noise = use_noise
        self.spectral_norm = spectral_norm

        # First convolution
        if use_noise: 
            self.conv_in = nn.Conv2d(nf_in, nf-1, kernel_size=9, stride=1, padding=4)
        else: 
            self.conv_in = nn.Conv2d(nf_in, nf, kernel_size=9, stride=1, padding=4)
        self.activation_in = nn.LeakyReLU(0.2)

        # Resblocks keeping shape
        self.resblocks = nn.Sequential(*[
            ResidualBlock(nf, spectral_norm=spectral_norm) for i in range(nres)
        ])
        # Resblocks with upscaling
        self.upblocks = nn.Sequential(*[
            UpsampleBlock(nf, spectral_norm=spectral_norm, method=upsample_method) for i in range(3)
        ])
        self.conv_out = nn.Conv2d(nf, nout, kernel_size=9, stride=1, padding=4)
        
        if spectral_norm: 
            self.conv_in = nn.utils.spectral_norm(self.conv_in)
            self.conv_out = nn.utils.spectral_norm(self.conv_out)
        
    def forward(self, x):
        out = self.conv_in(x)
        out = self.activation_in(out)
        if self.use_noise: 
            bs, _, h, w = x.shape
            z = torch.normal(0, 1, size=(bs, 1, h, w), device=device, requires_grad=True)
            out = torch.cat([out, z], dim=1)
        skip = out
        out = self.resblocks(out)
        out = out + skip
        out = self.upblocks(out)
        out = self.conv_out(out)
        if self.relu_out:
            out = nn.functional.relu(out)
        if self.softmax_out:
            out = nn.functional.softmax(out, dim=1)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, nf_in, nf_out, stride, batch_norm=True, spectral_norm=False):
        super().__init__()
        layers = []
        convlayer = nn.Conv2d(nf_in, nf_out, kernel_size=3, stride=stride, padding=1)
        if spectral_norm: 
            convlayer = nn.utils.spectral_norm(convlayer)
        layers.append(convlayer)
        if batch_norm:
            layers.append(nn.BatchNorm2d(nf_out))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module): 
    """ A first simple discriminator with binary output (sigmoid at final layer)"""
    def __init__(self, nfs, batch_norm=True, in_size=128, sigmoid=False, conditional=True, 
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
        self.sigmoid = sigmoid
        self.conditional = conditional
        self.spectral_norm = spectral_norm
        
        if self.conditional: 
            self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
            nf0 = 2
        else: 
            nf0 =1
            
        # First layer
        conv1 = nn.Conv2d(nf0, nfs[0], kernel_size=3, padding=1)
        if spectral_norm:
            conv1 = torch.nn.utils.spectral_norm(conv1)
        self.first_layer = nn.Sequential(
            conv1,
            nn.LeakyReLU(0.2),
            DiscriminatorBlock(
                nfs[0], nfs[0], stride=2, batch_norm=batch_norm, spectral_norm=spectral_norm
                )
        )
        
        # Intermediate layers
        int_layers = []
        for nf_in, nf_out in zip(nfs[:-1], nfs[1:]):
            int_layers.extend([
                DiscriminatorBlock(
                    nf_in, nf_out, stride=1, batch_norm=batch_norm, spectral_norm=spectral_norm
                    ), 
                DiscriminatorBlock(
                    nf_out, nf_out, stride=2, batch_norm=batch_norm, spectral_norm=spectral_norm
                    ),
            ])
        self.int_layers = nn.Sequential(*int_layers)
        
        # Final layers
        out_size = (in_size // 2**len(nfs))**2 * nfs[-1]
        print('Size after convolutions', out_size)
        linear1 = nn.Linear(out_size, 256)
        linear2 = nn.Linear(256, 1)
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
        out = self.first_layer(x)
        out = self.int_layers(out)
        out = out.view(out.shape[0], -1)   # Flatten
        out = self.final_layers(out)
        if self.sigmoid:
            out = nn.functional.sigmoid(out)
        return out
