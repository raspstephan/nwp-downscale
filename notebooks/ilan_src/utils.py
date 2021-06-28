def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

# if isnotebook():
#     print('In notebook')
#     import tqdm.notebook as tqdm
# else:
#     print('Not in notebook')
#     import tqdm
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import numpy as np

def plot_sample_old(X, y, gen, i=0):
    preds = gen(X.to(device)).detach().cpu().numpy()
    lr = X[i, 0].detach().cpu().numpy()
    hr = y[i, 0].detach().cpu().numpy()
    pred = preds[i, 0]
    
    mn = np.min([np.min(lr), np.min(hr), np.min(pred)])
    mx = np.max([np.max(lr), np.max(hr), np.max(pred)])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax1.imshow(lr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
    plt.colorbar(im, ax=ax1, shrink=0.7)
    im = ax2.imshow(pred, vmin=mn, vmax=mx, cmap='gist_ncar_r')
    plt.colorbar(im, ax=ax2, shrink=0.7)
    im = ax3.imshow(hr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
    plt.colorbar(im, ax=ax3, shrink=0.7)

    
def plot_sample(noise, cond, target, gen, k=1):
    with torch.no_grad():
        preds = gen(cond, noise).detach().cpu().numpy()
    
    for i in range(k):
        lr = cond[i, 0].detach().cpu().numpy()
        hr = target[i, 0].detach().cpu().numpy()
        pred = preds[i, 0]

        mn = np.min([np.min(hr), np.min(pred)])
        mx = np.max([np.max(hr), np.max(pred)])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im = ax1.imshow(lr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
        plt.colorbar(im, ax=ax1, shrink=0.7)
        im = ax2.imshow(pred, vmin=mn, vmax=mx, cmap='gist_ncar_r')
        plt.colorbar(im, ax=ax2, shrink=0.7)
        im = ax3.imshow(hr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
        plt.colorbar(im, ax=ax3, shrink=0.7)
        plt.show() 
        
        
def plot_samples_per_input(cond, target, gen, k=1, samples = 3):
    fig, axs = plt.subplots(k, samples+2, figsize=(15, k*5))
    gen_images = np.zeros((k,samples+2,128,128))
    with torch.no_grad():    
        for i in range(4):
            noise = torch.randn(cond.shape[0], 1, cond.shape[2], cond.shape[3])
            pred = gen(cond, noise).detach().cpu().numpy()
            for j in range(k):
                gen_images[j,i,:,:] = pred[j, 0] 

    for j in range(k):
        lr = cond[j, 0].detach().cpu().numpy()
        hr = target[j, 0].detach().cpu().numpy()
        mn = np.min([np.min(hr), np.min(pred), np.min(gen_images[j,i,:,:])])
        mx = np.max([np.max(hr), np.max(pred), np.max(gen_images[j,i,:,:])])
        im = axs[j,0].imshow(lr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
#         plt.colorbar(im, ax=axs[j,0], shrink=0.7)
        im = axs[j,1].imshow(hr, vmin=mn, vmax=mx, cmap='gist_ncar_r')
#         plt.colorbar(im, ax=axs[j,0], shrink=0.7)
        for i in range(samples):
            im = axs[j,i+2].imshow(gen_images[j,i,:,:], vmin=mn, vmax=mx, cmap='gist_ncar_r')
#             plt.colorbar(im, ax=axs[j,i], shrink=0.7)
    plt.show()  
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def to_categorical(y, num_classes=None, dtype='float32'):
    """Copied from keras source code
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical