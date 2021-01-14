import torch
import torch.nn as nn
from .utils import tqdm, device
import numpy as np
import matplotlib.pyplot as plt


class UpscalingCNN(nn.Module):
    """Totally untested"""
    def __init__(self, input_vars=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_vars, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)
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