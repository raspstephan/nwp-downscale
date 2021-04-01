import torch
import torch.nn as nn
from .utils import tqdm, device, plot_sample
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

            if (self.epoch-1) % self.valid_every_epochs == 0 and self.dl_valid:
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
def get_gradient(crit, X, real, fake, epsilon):
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
    mixed_scores = crit([X, mixed_images])
    
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

        
class GANTrainer():
    """Implements a Keras-style fit function and tracks train/valid losses"""
    def __init__(self, gen, disc, gen_optimizer, disc_optimizer,  dl_train, dl_valid=None, 
                 valid_every_epochs=1, disc_repeats=1, gen_repeats=1,
                 gp_lambda=None, l_loss=None, l_lambda=20, adv_loss_type='hinge',
                 save_dir=None, plot=False, plotting_sample=None):
        """ GAN trainer that includes options to set e.g. hinge loss, gradient penalty, ... 
        """
        self.gen = gen
        self.disc = disc
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.valid_every_epochs = valid_every_epochs
        self.adv_loss_type = adv_loss_type
        self.criterion = nn.BCELoss()
        if l_loss == 'l1':
            self.l_loss = nn.L1Loss()
        elif l_loss == 'l2':
            self.l_loss = nn.MSELoss()
        else:
            self.l_loss = None
        self.l_lambda = l_lambda
        self.disc_repeats = disc_repeats
        self.gen_repeats = gen_repeats
        self.gp_lambda = gp_lambda
        self.save_dir = save_dir
        self.plot = plot
        self.plotting_sample = plotting_sample
        
        self.epoch = 0
        self.train_gen_losses = []
        self.train_mse = []
        self.train_disc_losses = []
        self.train_epochs = []
        self.disc_preds_real = []
        self.disc_preds_fake = []
        self.gen_preds_fake = []
#         self.valid_losses = []
#         self.valid_epochs = []

    def _disc_loss(preds_real, preds_fake):
        """Returns adversarial loss for discriminator"""
        if self.adv_loss_type == 'hinge':  
            disc_loss = (
                nn.functional.relu(1 - torch.mean(preds_real)) + 
                nn.functional.relu(1 + torch.mean(preds_fake))
            )
        elif self.adv_loss_type == 'Wasserstein': 
            disc_loss =  -torch.mean(preds_real) + torch.mean(preds_fake)
        elif self.adv_loss_type == 'mse':
            ground_truth_real = torch.ones_like(preds_real)
            ground_truth_fake = torch.zeros_like(preds_fake)
            disc_loss_real = nn.functional.mse_loss(preds_real, ground_truth_real)
            disc_loss_fake = nn.functional.mse_loss(preds_fake, ground_truth_fake)
            disc_loss = disc_loss_real + disc_loss_fake
        return disc_loss
    
    def _gen_loss(preds_fake):
        """Returns adversarial loss for generator"""
        if self.adv_loss_type == 'mse':
            ground_truth = torch.ones_like(preds_fake)
            gen_loss = nn.functional.mse_loss(preds_fake, ground_truth)
        else:  # Wasserstein or Hinge
            gen_loss = -torch.mean(preds_fake)
        return gen_loss

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
                    preds_real = self.disc([X, real])
                    preds_fake = self.disc([X, fake.detach()])
                    self.disc_preds_real.append(preds_real.detach().cpu().numpy())
                    self.disc_preds_fake.append(preds_fake.detach().cpu().numpy())
                    
                    disc_loss = self._disc_loss(preds_real, preds_fake)
                    
                    if self.gp_lambda:
                        epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                        gradient = get_gradient(disc, X, real, fake.detach(), epsilon)
                        gp = gradient_penalty(gradient)
                        disc_loss += self.gp_lambda * gp
                    
                    mean_disc_loss += disc_loss.item() / self.disc_repeats
                    disc_loss.backward(retain_graph=True)
                    self.disc_optimizer.step()
                
                
                # Train generator
                mean_gen_loss = 0
                mean_mse = 0
                mean_l_loss = 0
                for _ in range(self.gen_repeats):
                    self.gen_optimizer.zero_grad()

                    fake = self.gen(X)
                    preds_fake = self.disc([X, fake])
                    self.gen_preds_fake.append(preds_fake.detach().cpu().numpy())

                    mse = nn.MSELoss()(fake, real).item()   # For diagnostics only

                    gen_loss = self._gen_loss(preds_fake)
                    if self.l_loss:
                        l_loss = self.l_loss(fake, real)
                        gen_loss += self.l_lambda * l_loss
                    gen_loss.backward()
                    self.gen_optimizer.step()
                    
                    mean_gen_loss += gen_loss.item() / self.gen_repeats
                    mean_mse += mse / self.gen_repeats
                    mean_l_loss += l_loss.item() / self.gen_repeats
               
                prog_bar.update()
#                 train_gen_loss += (loss.item() - train_loss) / (i+1)
                self.train_gen_losses.append(mean_gen_loss)
                self.train_mse.append(mse)
                self.train_disc_losses.append(mean_disc_loss)
                postfix = {
                    'train_gen_loss': mean_gen_loss,
                    'train_disc_loss': mean_disc_loss,
                    'mse': mean_mse,
                }
                if self.l_loss: postfix['l_loss'] = mean_l_loss
                prog_bar.set_postfix(postfix)
            self.train_epochs.append(self.epoch)

            if self.save_dir:
                torch.save(gen.state_dict(), f'{self.save_dir}/gen_{epoch}.pt')
                torch.save(disc.state_dict(), f'{self.save_dir}/disc_{epoch}.pt')
            if self.plot:
                if self.plotting_sample is None:
                    X_sample, y_sample = X, y
                else:
                    X_sample, y_sample = self.plotting_sample
                plot_sample(X_sample, y_sample, self.gen)
                plt.show()
            self.epoch += 1
        
    # def plot_losses(self, plot_valid=True):
    #     plt.plot(self.train_epochs, self.train_losses, label='Train')
    #     if plot_valid: plt.plot(self.valid_epochs, self.valid_losses, label='Valid')
    #     plt.legend()