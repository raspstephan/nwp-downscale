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


class Trainer():
    """Implements a Keras-style fit function and tracks train/valid losses"""
    def __init__(self, model, optimizer, criterion, dl_train, dl_valid=None, 
                 valid_every_epochs=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.valid_every_epochs = valid_every_epochs
        
        self.epoch = 0
        self.train_losses = []
        self.train_epochs = []
        self.valid_losses = []
        self.valid_epochs = []
        
    def fit(self, epochs):

        # Epoch loop
        for epoch in range(1, epochs+1):

            prog_bar = tqdm.tqdm(total=len(self.dl_train), desc=f'Epoch {epoch}')
            train_loss, valid_loss = 0, 0

            # Train
            for i, (X, y) in enumerate(self.dl_train):
                X = X.to(device); y = y.to(device)
                y_hat = self.model(X)
                loss = self.criterion(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                prog_bar.update()
                train_loss += (loss.cpu().detach().numpy() - train_loss) / (i+1)
                prog_bar.set_postfix({'train_loss': train_loss})
            self.train_losses.append(train_loss)
            self.train_epochs.append(self.epoch)

            if (self.epoch-1) % self.valid_every_epochs == 0:
                # Valid
                for i, (X, y) in enumerate(self.dl_valid):
                    X = X.to(device); y = y.to(device)
                    y_hat = self.model(X)
                    loss = self.criterion(y_hat, y)

                    valid_loss += (loss.cpu().detach().numpy() - valid_loss) / (i+1)
                self.valid_losses.append(valid_loss)
                self.valid_epochs.append(self.epoch)

                prog_bar.set_postfix({'train_loss': train_loss, 'valid_loss': valid_loss}) 
                prog_bar.close()
        
            self.epoch += 1
        
    def plot_losses(self, plot_valid=True):
        plt.plot(self.train_epochs, self.train_losses, label='Train')
        if plot_valid: plt.plot(self.valid_epochs, self.valid_losses, label='Valid')
        plt.legend()