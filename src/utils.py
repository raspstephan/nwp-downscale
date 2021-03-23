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

def plot_sample(X, y, gen, i=0):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)