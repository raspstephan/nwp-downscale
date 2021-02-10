import torch
import torch.nn as nn
from .utils import tqdm, device
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

class Trainer():
    """Implements a Keras-style fit function and tracks train/valid losses"""
    def __init__(self, model, optimizer, criterion, dl_train, dl_valid=None, 
                 valid_every_epochs=1, early_stopping_patience=None,
                 restore_best_weights=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.valid_every_epochs = valid_every_epochs
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self.id = datetime.now().isoformat()    # Unique identified for tmp early stopping weights
        if self.early_stopping_patience:
            assert self.valid_every_epochs == 1, 'Must validate every epoch for early stopping'
        
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

            if self.early_stopping_patience:
                stop = self._check_early_stopping()
                if stop:
                    print('Early stopping triggered!')
                    break
        
    def _check_early_stopping(self):
        # Is last valid loss best valid loss
        min_epoch = np.argmin(self.valid_losses) + 1
        os.makedirs('./tmp', exist_ok=True)
        tmp_path = f'./tmp/{self.id}.pt'

        if min_epoch == self.epoch and self.restore_best_weights:
            torch.save(self.model.state_dict(), tmp_path)
        
        # No improvement in patience
        if (self.epoch - min_epoch) == self.early_stopping_patience:
            if self.restore_best_weights:
                self.model.load_state_dict(torch.load(tmp_path))
            return True
        else:
            return False
        

    def plot_losses(self, plot_valid=True):
        plt.plot(self.train_epochs, self.train_losses, label='Train')
        if plot_valid: plt.plot(self.valid_epochs, self.valid_losses, label='Valid')
        plt.legend()