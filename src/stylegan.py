import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
import torchvision
from pathlib import Path
from PIL import Image

class MappingNetwork(nn.Module):
    """
    <a id="mapping_network"></a>
    ## Mapping Network
    ![Mapping Network](mapping_network.svg)
    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)


class Generator(nn.Module):
    """
    <a id="generator"></a>
    ## StyleGAN2 Generator
    ![Generator](style_gan2.svg)
    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [*toRGB*](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.</small>*
    The generator starts with a learned constant.
    Then it has a series of blocks. The feature map resolution is doubled at each block
    Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.
    """

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `d_latent` is the dimensionality of $w$
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Calculate the number of features for each block
        #
        # Something like `[512, 512, 256, 128, 64, 32]`
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        # Number of generator blocks
        self.n_blocks = len(features)

        # Trainable $4 \times 4$ constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # First style block for $4 \times 4$ resolution and layer to get RGB
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_greyscale = ToGreyScale(d_latent, features[0])

        # Generator blocks
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # $2 \times$ up sampling layer. The feature space is up sampled
        # at each block
        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        * `w` is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate
        $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]1.
        * `input_noise` is the noise for each block.
        It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs
        after each convolution layer (see the diagram).
        """

        # Get batch size
        batch_size = w.shape[1]

        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # The first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Get first rgb image
        greyscale = self.to_greyscale(x, w[0])

        # Evaluate rest of the blocks
        for i in range(1, self.n_blocks):
            # Up sample the feature map
            x = self.up_sample(x)
            # Run it through the [generator block](#generator_block)
            x, greyscale_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # Up sample the RGB image and add to the rgb from the block
            greyscale = self.up_sample(greyscale) + greyscale_new

        # Return the final RGB image
        return greyscale


class GeneratorBlock(nn.Module):
    """
    <a id="generator_block"></a>
    ### Generator Block
    ![Generator block](generator_block.svg)
    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [*toRGB*](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.</small>*
    The generator block consists of two [style blocks](#style_block) ($3 \times 3$ convolutions with style modulation)
    and an RGB output.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        # Second [style block](#style_block)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        # *toRGB* layer
        self.to_greyscale = ToGreyScale(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])

        # Get greyscale image
        greyscale = self.to_greyscale(x, w)

        # Return feature map and rgb image
        return x, greyscale


class StyleBlock(nn.Module):
    """
    <a id="style_block"></a>
    ### Style Block
    ![Style block](style_block.svg)
    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is single channel).</small>*
    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tensor of shape `[batch_size, 1, height, width]`
        """
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    <a id="to_rgb"></a>
    ### To RGB
    ![To RGB](to_rgb.svg)
    *<small>$A$ denotes a linear layer.</small>*
    Generates an RGB image from a feature map using $1 \times 1$ convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])

class ToGreyScale(nn.Module):
    """
    <a id="to_rgb"></a>
    ### To RGB
    ![To RGB](to_rgb.svg)
    *<small>$A$ denotes a linear layer.</small>*
    Generates an RGB image from a feature map using $1 \times 1$ convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 1, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])

class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation
    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-5):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)
    
class Conv2dWeightModulateNoStyle(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation
    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-5):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        self.in_features = in_features
        self.kernel_size = kernel_size
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape
        
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight().unsqueeze(0).expand(b,self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    """
    <a id="discriminator"></a>
    ## StyleGAN 2 Discriminator
    ![Discriminator](style_gan2_disc.svg)
    Discriminator first transforms the image to a feature map of the same resolution and then
    runs it through a series of blocks with residual connections.
    The resolution is down-sampled by $2 \times$ at each block while doubling the
    number of features.
    """

    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Layer to convert greyscae image to a feature map with `n_features` number of features.
        self.from_greyscale = nn.Sequential(
            EqualizedConv2d(1, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        # Calculate the number of features for each block.
        #
        # Something like `[64, 128, 256, 512, 512, 512]`.
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # Number of [discirminator blocks](#discriminator_block)
        n_blocks = len(features) - 1
        # Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # [Mini-batch Standard Deviation](#mini_batch_std_dev)
        self.std_dev = MiniBatchStdDev()
        # Number of features after adding the standard deviations map
        final_features = features[-1] + 1
        # Final $3 \times 3$ convolution layer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # Final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, 3, height, width]`
        """

        # Try to normalize the image (this is totally optional, but sped up the early training a little)
        x = x - 0.5
#         print(x.shape)
        # Convert from RGB
        x = self.from_greyscale(x)
        # Run through the [discriminator blocks](#discriminator_block)
        x = self.blocks(x)

        # Calculate and append [mini-batch standard deviation](#mini_batch_std_dev)
        x = self.std_dev(x)
        # $3 \times 3$ convolution
        x = self.conv(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
#         print(x.shape)
        # Return the classification score
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    """
    <a id="discriminator_black"></a>
    ### Discriminator Block
    ![Discriminator block](discriminator_block.svg)
    Discriminator block consists of two $3 \times 3$ convolutions with a residual connection.
    """

    def __init__(self, in_features, out_features):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Down-sampling and $1 \times 1$ convolution layer for the residual connection
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1, padding=0))

        # Two $3 \times 3$ convolutions
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Down-sampling layer
        self.down_sample = DownSample()

        # Scaling factor $\frac{1}{\sqrt 2}$ after adding the residual
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # Get the residual connection
        residual = self.residual(x)

        # Convolutions
        x = self.block(x)
        # Down-sample
        x = self.down_sample(x)

        # Add the residual and scale
        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """
    <a id="mini_batch_std_dev"></a>
    ### Mini-batch Standard Deviation
    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        # $$\mu_{i} = \frac{1}{N} \sum_g x_{g,i} \\
        #   \sigma_{i} = \sqrt{\frac{1}{N} \sum_g (x_{g,i} - \mu_i)^2  + \epsilon}$$
        std = torch.sqrt(grouped.var(dim=0) + 1e-5)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    """
    <a id="down_sample"></a>
    ### Down-sample
    The down-sample operation [smoothens](#smooth) each feature channel and
     scale $2 \times$ using bilinear interpolation.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://papers.labml.ai/paper/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Smoothing or blurring
        x = self.smooth(x)
        # Scaled down
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    """
    <a id="up_sample"></a>
    ### Up-sample
    The up-sample operation scales the image up by $2 \times$ and [smoothens](#smooth) each feature channel.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://papers.labml.ai/paper/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    """
    <a id="smooth"></a>
    ### Smoothing Layer
    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>
    ## Learning-rate Equalized Linear Layer
    This uses [learning-rate equalized weights]($equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights]($equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    <a id="equalized_conv2d"></a>
    ## Learning-rate Equalized 2D Convolution Layer
    This uses [learning-rate equalized weights]($equalized_weights) for a convolution layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 1):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights]($equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>
    ## Learning-rate Equalized Weights Parameter
    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\mathcal{N}(0,c)$ they initialize weights
    to $\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \hat{w}_i$$
    The gradients on stored parameters $\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.
    The optimizer updates on $\hat{w}$ are proportionate to the learning rate $\lambda$.
    But the effective weights $w$ get updated proportionately to $c \lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\lambda$.
    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c


class GradientPenalty(nn.Module):
    """
    <a id="gradient_penalty"></a>
    ## Gradient Penalty
    This is the $R_1$ regularization penality from the paper
    [Which Training Methods for GANs do actually Converge?](https://papers.labml.ai/paper/1801.04406).
    $$R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{p_\mathcal{D}(x)}
    \Big[\Vert \nabla_x D_\psi(x)^2 \Vert\Big]$$
    That is we try to reduce the L2 norm of gradients of the discriminator with
    respect to images, for real images ($P_\mathcal{D}$).
    """

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        * `x` is $x \sim \mathcal{D}$
        * `d` is $D(x)$
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to $1$ since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        
        # Calculate the norm $\Vert \nabla_{x} D(x)^2 \Vert$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\Vert \nabla_x D_\psi(x)^2 \Vert$
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    """
    <a id="path_length_penalty"></a>
    ## Path Length Penalty
    This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude
    change in the image.
    $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
      \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
    where $\mathbf{J}_w$ is the Jacobian
    $\mathbf{J}_w = \frac{\partial g}{\partial w}$,
    $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and
    $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.
    $a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
    as the training progresses.
    $\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
    $$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant $\beta$ used to calculate the exponential moving average $a$
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # Exponential sum of $\mathbf{J}^\top_{w} y$
        # $$\sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
        # where $[\mathbf{J}^\top_{w} y]_i$ is the value of it at $i$-th step of training
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate $y \in \mathcal{N}(0, \mathbf{I})$
        y = torch.randn(x.shape, device=device)
        # Calculate $\big(g(w) \cdot y \big)$ and normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / math.sqrt(image_size)

        # Calculate gradients to get $\mathbf{J}^\top_{w} y$
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculate L2-norm of $\mathbf{J}^\top_{w} y$
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate $a$
            # $$\frac{1}{1 - \beta^N} \sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            # $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
            # \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment $N$
        self.steps.add_(1.)

        # Return the penalty
        return loss
    
    
    
class Dataset(torch.utils.data.Dataset):
    """
    ## Dataset
    This loads the training dataset and resize it to the give image size.
    """

    def __init__(self, path: str, image_size: int):
        """
        * `path` path to the folder containing the images
        * `image_size` size of the image
        """
        super().__init__()

        # Get the paths of all `jpg` files
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]
        
#         print(list(Path(path).glob(f'**/*.jpg')))
        
        # Transformation
        self.transform = torchvision.transforms.Compose([
            # Resize the image
            torchvision.transforms.Resize(image_size),
            # Convert to PyTorch tensor
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """Number of images"""
        return len(self.paths)

    def __getitem__(self, index):
        """Get the the `index`-th image"""
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img), self.transform(img)
    
    
class StyleGan2(LightningModule):
    def __init__(self): # fill in
        super().__init__()
        self.image_size = 128
        self.d_latent = 512
        log_resolution = int(math.log2(self.image_size))
        self.learning_rate = 1e-3
        self.mapping_network_learning_rate = 1e-5
        
        self.style_mixing_prob = 0.9
        
        # Create discriminator and generator
        self.discriminator = Discriminator(log_resolution, n_features = 32, max_features = 256)
        self.generator = Generator(log_resolution, self.d_latent,  
                                   n_features = 32, max_features = 256)
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network_layers = 6
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers)
 
        # Create path length penalty loss

        self.lazy_gradient_penalty_interval = 4
        self.batch_size = 32
        self.gradient_penalty = GradientPenalty()
        self.gradient_penalty_coefficient = 10.0
        self.path_length_penalty = PathLengthPenalty(0.99)
        self.lazy_path_penalty_interval = 32
        self.lazy_path_penalty_after = 5000
        
        self.automatic_optimization = False
#         self.gradient_accumulate_steps = 1
        
#         self.save_hyperparameters()
        
    def get_w(self, batch_size: int):
        """
        ### Sample $w$
        This samples $z$ randomly and get $w$ from the mapping network.
        We also apply style mixing sometimes where we generate two latent variables
        $z_1$ and $z_2$ and get corresponding $w_1$ and $w_2$.
        Then we randomly sample a cross-over point and apply $w_1$ to
        the generator blocks before the cross-over point and
        $w_2$ to the blocks after.
        """

        # Mix styles
        if torch.rand(()).item() < self.style_mixing_prob:
            # Random cross-over point
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            # Sample $z_1$ and $z_2$
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            # Get $w_1$ and $w_2$
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            # Expand $w_1$ and $w_2$ for the generator blocks and concatenate
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        # Without mixing
        else:
            # Sample $z$ and $z$
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            # Get $w$ and $w$
            w = self.mapping_network(z)
            # Expand $w$ for the generator blocks
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_noise(self, batch_size: int):
        """
        ### Generate noise
        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, batch_size: int):
        """
        ### Generate images
        This generate images using the generator
        """

        # Get $w$
        w = self.get_w(batch_size)
        # Get noise
        noise = self.get_noise(batch_size)

        # Generate images
        images = self.generator(w, noise)

        # Return images and $w$
        return images, w
    
    def configure_optimizers(self):
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0, 0.99), eps=1e-5)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0, 0.99), eps=1e-5)
        mapping_network_optimizer = optim.Adam(self.mapping_network.parameters(), lr=self.mapping_network_learning_rate, betas=(0, 0.99), eps=1e-5)
            
        return discriminator_optimizer, generator_optimizer, mapping_network_optimizer
    
    def forward(self):
        return self.gen(x,w,noise)
    
    
    def discriminator_loss(self, f_real, f_fake):
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()
    
    def generator_loss(self, f_fake):
        return -f_fake.mean()

    
    def training_step(self, batch, batch_idx):

        condition, real_images = batch[0], batch[1]
        
        discriminator_optimizer, generator_optimizer, mapping_network_optimizer = self.optimizers()
        
        # Sample images from generator
        generated_images, _ = self.generate_images(self.batch_size)
        # Discriminator classification for generated images
        fake_output = self.discriminator(generated_images.detach())

        # We need to calculate gradients w.r.t. real images for gradient penalty
        if (self.global_step + 1) % self.lazy_gradient_penalty_interval == 0:
            real_images.requires_grad_()
        # Discriminator classification for real images
        real_output = self.discriminator(real_images)
        
        print("nans in output:", torch.sum(torch.isnan(real_output)))
        
        # Get discriminator loss
        real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
        disc_loss = real_loss + fake_loss

        # Add gradient penalty
        if (self.global_step + 1) % self.lazy_gradient_penalty_interval == 0:
            # Calculate and log gradient penalty
            gp = self.gradient_penalty(real_images, real_output)
            # Multiply by coefficient and add gradient penalty
            print("gp:", gp)
            disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

        # Compute gradients
        discriminator_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        
#         grads = [par.grad for par in self.discriminator.parameters() if (par.grad is not None)]
#         max_grad = max([torch.max(g) for g in grads])
#         print("num params:", len(list(self.discriminator.parameters())))
#         print("number params with grad:", len(grads))
        
#         grad_param_shapes = [par.grad.shape for par in self.discriminator.parameters() if (par.grad is not None)]
#         print("grad param shapes:", grad_param_shapes)
#         print("max_grad:", max_grad)

                             
        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # Take optimizer step
        discriminator_optimizer.step()
    
        # Sample images from generator
        generated_images, w = self.generate_images(self.batch_size)
        # Discriminator classification for generated images
        fake_output = self.discriminator(generated_images)

        # Get generator loss
        gen_loss = self.generator_loss(fake_output)

        # Add path length penalty
        if self.global_step > self.lazy_path_penalty_after and (self.global_step + 1) % self.lazy_path_penalty_interval == 0:
            # Calculate path length penalty
            plp = self.path_length_penalty(w, generated_images)
            # Ignore if `nan`
            if not torch.isnan(plp):
                gen_loss = gen_loss + plp

        # Reset gradients
        generator_optimizer.zero_grad()
        mapping_network_optimizer.zero_grad()
        
        # Calculate gradients
        self.manual_backward(gen_loss)
        
        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

        # Take optimizer step
        generator_optimizer.step()
        mapping_network_optimizer.step()
        
        # Log loss
        
        self.log('discriminator_loss', disc_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log('generator_loss', gen_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        
        if self.global_step%50==0:
            self.generator.eval()
            # Sample images from generator
            generated_images, _ = self.generate_images(16)
#             print(generated_images.shape)
            grid = torchvision.utils.make_grid(torch.cat([generated_images, real_images[:16]], dim=0))
            self.logger.experiment.add_image('generated_images', grid, self.global_step)
            self.generator.train()
        
        
        
        
        
        
        
