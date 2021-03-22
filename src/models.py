import torch
import torch.nn as nn
from .utils import tqdm, device
import numpy as np
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x

class UpsampleBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf * 4, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.PixelShuffle(2)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        out = self.activation(out)
        return out

class Generator(nn.Module):
    def __init__(self, nres, nf_in, nf, relu_out=False):
        super().__init__()
        self.conv_in = nn.Conv2d(nf_in, nf, kernel_size=9, stride=1, padding=4)
        self.activation_in = nn.LeakyReLU(0.2)
        self.resblocks = nn.Sequential(*[
            ResidualBlock(nf) for i in range(nres)
        ])
        self.upblocks = nn.Sequential(*[
            UpsampleBlock(nf) for i in range(3)
        ])
        self.conv_out = nn.Conv2d(nf, 1, kernel_size=9, stride=1, padding=4)
        self.relu_out = relu_out
        
    def forward(self, x):
        out = self.conv_in(x)
        out = self.activation_in(out)
        skip = out
        out = self.resblocks(out)
        out = out + skip
        out = self.upblocks(out)
        out = self.conv_out(out)
        if self.relu_out:
            out = nn.functional.relu(out)
        return out

### Discriminator

class DiscriminatorBlock(nn.Module):
    def __init__(self, nf_in, nf_out, stride, batch_norm=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(nf_in, nf_out, kernel_size=3, stride=stride, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(nf_out))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, nfs, batch_norm=True, in_size=128, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
        
        # First layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, nfs[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            DiscriminatorBlock(nfs[0], nfs[0], stride=2, batch_norm=batch_norm)
        )
        
        # Intermediate layers
        int_layers = []
        for nf_in, nf_out in zip(nfs[:-1], nfs[1:]):
            int_layers.extend([
                DiscriminatorBlock(nf_in, nf_out, stride=1, batch_norm=batch_norm),
                DiscriminatorBlock(nf_out, nf_out, stride=2, batch_norm=batch_norm),
            ])
        self.int_layers = nn.Sequential(*int_layers)
        
        # Final layers
        out_size = (in_size // 2**len(nfs))**2 * nfs[-1]
        print('Size after convolutions', out_size)
        self.final_layers = nn.Sequential(
            nn.Linear(out_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
            
        
    def forward(self, x):
        out = self.first_layer(x)
        out = self.int_layers(out)
        out = out.view(out.shape[0], -1)   # Flatten
        out = self.final_layers(out)
        if self.sigmoid:
            out = torch.functional.sigmoid(out)
        return out
