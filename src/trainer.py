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



### GANs
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm -1)**2)
    #### END CODE HERE ####
    return penalty

class WGANTrainer():
    """Implements a Keras-style fit function and tracks train/valid losses"""
    def __init__(self, gen, disc, gen_optimizer, disc_optimizer,  dl_train, dl_valid=None, 
                 valid_every_epochs=1, disc_repeats=1, gp_lambda=10):
        self.gen = gen
        self.disc = disc
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.valid_every_epochs = valid_every_epochs
        self.criterion = nn.BCELoss()
        self.disc_repeats = disc_repeats
        self.gp_lambda = gp_lambda
        
        self.epoch = 0
        self.train_gen_losses = []
        self.train_mse = []
        self.train_disc_losses = []
        self.train_epochs = []
#         self.valid_losses = []
#         self.valid_epochs = []
        
    def fit(self, epochs):

        # Epoch loop
        for epoch in range(1, epochs+1):

            prog_bar = tqdm.tqdm(total=len(self.dl_train), desc=f'Epoch {epoch}')
            train_loss, valid_loss = 0, 0

            # Train
            for i, (X, y) in enumerate(self.dl_train):
                X = X.to(device); real = y.to(device)
                bs = X.shape[0]
                
                mean_disc_loss = 0
                for _ in range(self.disc_repeats):
                    # Train discriminator
                    self.disc_optimizer.zero_grad()

                    fake = self.gen(X)
                    preds_real = self.disc(real)
                    preds_fake = self.disc(fake.detach())
                    
                    epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                    gradient = get_gradient(disc, real, fake.detach(), epsilon)
                    gp = gradient_penalty(gradient)
                    
                    disc_loss = -torch.mean(preds_real) + torch.mean(preds_fake) + self.gp_lambda * gp

                    mean_disc_loss += disc_loss.item() / self.disc_repeats
                    disc_loss.backward(retain_graph=True)
                    self.disc_optimizer.step()
                
                
                # Train generator
                self.gen_optimizer.zero_grad()
                
                fake = self.gen(X)
                preds_fake = self.disc(fake)
                
                mse = nn.MSELoss()(fake, real).item()
                
                gen_loss = -torch.mean(preds_fake)
                gen_loss.backward()
                self.gen_optimizer.step()
                

                prog_bar.update()
#                 train_gen_loss += (loss.item() - train_loss) / (i+1)
                self.train_gen_losses.append(gen_loss.item())
                self.train_mse.append(mse)
                self.train_disc_losses.append(mean_disc_loss)
                prog_bar.set_postfix({
                    'train_gen_loss': gen_loss.item(),
                    'train_disc_loss': disc_loss.item(),
                    'mse': mse
                })
#             self.train_gen_losses.append(gen_loss)
            self.train_epochs.append(self.epoch)

#             if (self.epoch-1) % self.valid_every_epochs == 0:
#                 # Valid
#                 for i, (X, y) in enumerate(self.dl_valid):
#                     X = X.to(device); y = y.to(device)
#                     y_hat = self.model(X)
#                     loss = self.criterion(y_hat, y)

#                     valid_loss += (loss.cpu().detach().numpy() - valid_loss) / (i+1)
#                 self.valid_losses.append(valid_loss)
#                 self.valid_epochs.append(self.epoch)

#                 prog_bar.set_postfix({'train_loss': train_loss, 'valid_loss': valid_loss}) 
#                 prog_bar.close()
        
            self.epoch += 1
        
    def plot_losses(self, plot_valid=True):
        plt.plot(self.train_epochs, self.train_losses, label='Train')
        if plot_valid: plt.plot(self.valid_epochs, self.valid_losses, label='Valid')
        plt.legend()