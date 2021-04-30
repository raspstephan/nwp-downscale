import torch
import torch.nn as nn


def make_conv2d(in_channels, out_channels, kernel_size, spectral_norm=False, **kwargs):
    """Convenience constructor for conv2d with and without spectral norm"""
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    if spectral_norm:
        conv2d = nn.utils.spectral_norm(conv2d)
    return conv2d


class ResidualBlock(nn.Module):
    def __init__(self, nf_in, nf_out, stride=1, spectral_norm=False, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm

        self.activation1 = nn.LeakyReLU(0.2)
        self.conv1 = make_conv2d(
            nf_in, nf_out, kernel_size=3, padding=1, stride=stride, padding_mode='reflect', spectral_norm=spectral_norm
            )
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(nf_out)

        self.activation2 = nn.LeakyReLU(0.2)
        self.conv2 = make_conv2d(
            nf_out, nf_out, kernel_size=3, padding=1, padding_mode='reflect', spectral_norm=spectral_norm
            )
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(nf_out)

        if nf_in != nf_out:
            self.skip_conv = make_conv2d(nf_in, nf_out, kernel_size=1, spectral_norm=spectral_norm)
            if stride > 1:
                self.skip_conv = nn.Sequential(
                    nn.AvgPool2d(stride),
                    self.skip_conv
                )
    
    def forward(self, x):
        out = self.activation1(x)
        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(x)
        return out + x


class UpsampleBlock(nn.Module):
    def __init__(self, nf_in, nf_out, spectral_norm=False, method='bilinear', batch_norm=False):
        super().__init__()
        # if method == 'PixelShuffle':
        #     self.upsample = nn.PixelShuffle(2)
        if method == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            raise NotImplementedError
        self.resblock = ResidualBlock(nf_in, nf_out, spectral_norm=spectral_norm, batch_norm=batch_norm)
        
    def forward(self, x):
        out = self.upsample(x)
        out = self.resblock(out)
        return out