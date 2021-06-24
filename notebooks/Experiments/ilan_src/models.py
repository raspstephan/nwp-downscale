import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchvision

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size*img_size)
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img+1, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, labels, x):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):
    def __init__(self, noise_shape, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(noise_shape[0] + embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding  = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim = 1)
        x = self.net(x)
#         print(x.shape)
        return x

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

#####################################################
####################################################

class DSDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(DSDiscriminator, self).__init__()
        
        self.img_size = img_size
        self.embed = nn.Sequential(
                        # Input: N x channels_noise x 16 x 16
                        self._expand_block(channels_img, features_d * 16, 1, 1, 0),  # img: 16x16
                        self._expand_block(features_d * 16, features_d * 8, 4, 2, 1),  # img: 32x32
                        self._expand_block(features_d * 8, features_d * 4, 4, 2, 1),  # img: 64x64
                        nn.ConvTranspose2d(
                            features_d * 4, channels_img, kernel_size=4, stride=2, padding=1
                        ),
                        # Output: N x channels_img x 128 x 128
                        nn.Sigmoid(),
                    )
        self.disc = nn.Sequential(
            # input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img*2, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 6, 4, 2, 1),
            self._block(features_d * 6, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def _expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


class DSGenerator(nn.Module):
    def __init__(self, noise_shape, channels_img, features_g, num_classes, img_size, embed_size):
        super(DSGenerator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Conv2d(in_channels=1, out_channels=embed_size, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            # Input: N x channels_noise x 16 x 16
            self._block(noise_shape[0] + embed_size, features_g * 16, 1, 1, 0),  # img: 16x16
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 32x32
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 64x64
            nn.ConvTranspose2d(
                features_g * 4, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 128 x 128
            nn.Sigmoid(),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, labels, x):
        embedding  = self.embed(labels)
        x = torch.cat([x, embedding], dim = 1)
        x = self.net(x)
#         print(x.shape)
        return x

    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                
                
                
###################################################################
###################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=256, planes=256, stride=1, nonlin = 'relu'):
        super(BasicBlock, self).__init__()
        self.nonlin = nonlin
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.nonlin = nonlin

    def forward(self, x):
        if self.nonlin == 'leaky_relu':
            out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.02)
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.leaky_relu(out, negative_slope=0.02)
            return out
        elif self.nonlin == 'relu':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out
        else: raise NotImplementedError
            
class ConvBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size = 3, norm=None, stride=1, activation='leaky_relu', padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding,
                stride=stride)
        self.norm = norm
        self.activation = activation
        
    def forward(self, x):
        
        if self.norm=="batch":
            x = nn.BatchNorm2d(in_channels)(x)
        if self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.2)
        elif self.activation == 'relu':
            x = F.relu(x)
        x = self.conv(x)
        return x

        return block 
    
class LeinResBlock(nn.Module):

    def __init__(self, in_planes=256, planes=256, stride=1, nonlin = 'relu', norm =None):
        super(LeinResBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.nonlin = nonlin
        self.norm = norm
        
        shortcut_modules = []
        if self.stride>1:
            shortcut_modules.append(nn.AvgPool2d(self.stride))
        if (self.planes != self.in_planes):
                shortcut_modules.append(ConvBlock(self.in_planes, self.planes, 1, stride=1, 
                    activation=False, padding=0))
        
        self.shortcut = nn.Sequential(*shortcut_modules)   
        self.convblock1 = ConvBlock(self.in_planes, self.planes, 3, stride=self.stride,
            norm=self.norm,
            activation=self.nonlin)
        self.convblock2 = ConvBlock(self.planes, self.planes, 3, stride=1,
            norm=self.norm,
            activation=self.nonlin)
        
    def forward(self, x):
        x_in = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x_in = self.shortcut(x_in)
        x = x + x_in
        return x

    
            

            
class DeconvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=256, planes=256, stride=2, nonlin = 'relu'):
        super(DeconvBlock, self).__init__()
        self.nonlin = nonlin
        self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.nonlin = nonlin


    def forward(self, x):
        if self.nonlin == 'leaky_relu':
            out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.02)
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.leaky_relu(out, negative_slope=0.02)
            return out
        elif self.nonlin == 'relu':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out
        else: raise NotImplementedError
            

class UpSample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(UpSample, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
#         print(x.shape)
        return x
    
    
class LeinGen(nn.Module):
    def __init__(self):
        super(LeinGen, self).__init__()
        self.embed = nn.Conv2d(1,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'), 
                                      LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'), 
                            #         self.b3 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=256, planes=128, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=128, planes=64, stride=1,  nonlin = 'leaky_relu'),
                                     UpSample(2, 'bilinear'),
                                     LeinResBlock(in_planes=64, planes=32, stride=1,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
#         print(x.shape)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
                                     
class LeinDisc(nn.Module):
    def __init__(self, nonlin = 'leaky_relu'):
        super(LeinDisc, self).__init__()
        hr_block = []
        lr_block = []
        inplanes = 1
        for planes in [64, 128, 256]:
            hr_block.append(LeinResBlock(in_planes = inplanes, planes=planes, stride=2, nonlin = nonlin))
            lr_block.append(LeinResBlock(in_planes = inplanes, planes=planes, stride=1, nonlin = nonlin))
            inplanes=planes
        self.hr_block1 = nn.Sequential(*hr_block)
        self.lr_block1 = nn.Sequential(*lr_block)
        self.hr_block2 = nn.Sequential(LeinResBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(LeinResBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    
#     def spectral_norm(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 m = nn.utils.parametrizations.spectral_norm(m)
    
    
                                     
                                     
class LeinGen2(nn.Module):
    def __init__(self):
        super(LeinGen, self).__init__()
        self.embed = nn.Conv2d(1,255, kernel_size=3, padding=1)
        self.process = nn.Sequential(BasicBlock(in_planes=256, planes=256, stride=1,  nonlin = 'relu'), 
                                      BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu'), 
                            #         self.b3 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'relu')
                            #         self.b4 = BasicBlock(in_planes=256, planes=256, stride=1, nonlin = 'leaky_relu')
                                        )
        self.upscale = nn.Sequential(DeconvBlock(in_planes=256, planes=128, stride=2, nonlin = 'leaky_relu'), 
                                     DeconvBlock(in_planes=128, planes=64, stride=2,  nonlin = 'leaky_relu'), 
                                     DeconvBlock(in_planes=64, planes=32, stride=2,  nonlin = 'leaky_relu'))
        
        self.final = nn.Conv2d(32,1, kernel_size=3, padding=1)
         
    def forward(self, x, noise):
        x = F.relu(self.embed(x))
        x = torch.cat((x,noise), axis=1)
        x = self.process(x)
        x = self.upscale(x)
        x = torch.sigmoid(self.final(x))
        return x
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
    
class LeinDisc2(nn.Module):
    def __init__(self, nonlin = 'leaky_relu'):
        super(LeinDisc, self).__init__()
        hr_block = []
        lr_block = []
        inplanes = 1
        for planes in [64, 128, 256]:
            hr_block.append(BasicBlock(in_planes = inplanes, planes=planes, stride=2, nonlin = nonlin))
            lr_block.append(BasicBlock(in_planes = inplanes, planes=planes, stride=1, nonlin = nonlin))
            inplanes=planes
        self.hr_block1 = nn.Sequential(*hr_block)
        self.lr_block1 = nn.Sequential(*lr_block)
        self.hr_block2 = nn.Sequential(BasicBlock(in_planes=256, planes=256, stride=1, nonlin = nonlin))#, block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.lr_block2 = nn.Sequential(BasicBlock(in_planes=512, planes=256, stride=1, nonlin = nonlin))#,block(in_planes=256, planes=256, stride=1, nonlin = nonlin))
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity = 'linear')
        self.initialize_weights()
        
        

    def forward(self, X, y):
        hr = self.hr_block1(y)
        lr = self.lr_block1(X)
#         print(lr.shape)
#         print(hr.shape)
        lr = torch.cat((lr,hr), axis=1)
        hr = self.hr_block2(hr)
        lr = self.lr_block2(lr)
        hr = nn.AvgPool2d(16)(hr)
        lr = nn.AvgPool2d(16)(lr)
        out = torch.cat((torch.squeeze(hr), torch.squeeze(lr)), axis=1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.02)
        out = self.dense2(out)
        return out
    
    def initialize_weights(self):
        # Initializes weights according to the DCGAN paper
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):#, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data)
            
    
    
#########################################################
##########################################################

class WGANGP(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, channels_img, features_g, num_classes, img_size, embed_size, features_d, 
                      b1 = 0.0, b2 = 0.9, lr = 1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1): # fill in
        super().__init__()
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.disc_freq, self.gen_freq = 5, 1
        self.noise_shape = noise_shape
        self.lambda_gp = lambda_gp
        self.gen = generator(noise_shape, channels_img, features_g, num_classes, img_size, embed_size)
        self.disc = discriminator(channels_img, features_d, num_classes, img_size)
        self.num_to_visualise = 12
        self.num_classes = num_classes
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%100==0:
            with torch.no_grad():
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
#                 print(sample_imgs.shape)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                
        
#         # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            fake = self.gen(condition, noise)
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake).reshape(-1)
            gp = self.gradient_penalty(condition, real, fake)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.lambda_gp*gp
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc

        
#         #train generator
        elif optimizer_idx ==1:
            noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            fake = self.gen(condition, noise)
            gen_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        disc_opt = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]


#########################################
# If train disc and gen on same batch:
#######################################


# class WGANGP(LightningModule):
#     def __init__(self, generator, discriminator, noise_shape, channels_img, features_g, num_classes, img_size, embed_size, features_d, 
#                       b1 = 0.0, b2 = 0.9, lr = 1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1): # fill in
#         super().__init__()
#         self.lr, self.b1, self.b2 = lr, b1, b2
#         self.disc_freq, self.gen_freq = 5, 1
#         self.noise_shape = noise_shape
#         self.lambda_gp = lambda_gp
#         self.gen = generator(noise_shape, channels_img, features_g, num_classes, img_size, embed_size)
#         self.disc = discriminator(channels_img, features_d, num_classes, img_size)
#         self.num_to_visualise = 12
#         self.num_classes = num_classes
#         self.automatic_optimization = False
#         self.real_idx = real_idx
#         self.cond_idx = cond_idx
        
        
#     def forward(self, condition, noise):
#         return self.gen(condition, noise)
    
#     def gradient_penalty(self, condition, real, fake):
#         BATCH_SIZE, C, H, W = real.shape
#         epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
#         interpolated_images = real*epsilon + fake*(1-epsilon)
#         mixed_scores = self.disc(condition, interpolated_images)
#         gradient = torch.autograd.grad(
#                     inputs=interpolated_images,
#                     outputs=mixed_scores, 
#                     grad_outputs = torch.ones_like(mixed_scores), 
#                     create_graph=True, 
#                     retain_graph = True)[0]

#         gradient = gradient.view(gradient.shape[0], -1)
#         gradient_norm = gradient.norm(2, dim=1)
#         gradient_penalty = torch.mean((gradient_norm - 1)**2)
#         return gradient_penalty
    
# #     def training_step(self, batch, batch_idx, optimizer_idx):
#     def training_step(self, batch, batch_idx):
#         gen_opt, disc_opt = self.optimizers()
        
# #         real, condition = batch # if label is condition.
#         condition, real = batch[self.cond_idx], batch[self.real_idx]
# #         print(real.device)
# #         print(self.global_step)
#         if self.global_step%100==0:
#             with torch.no_grad():
#                 noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
#         #         # log sampled images
#                 sample_imgs = self.gen(condition, noise)
# #                 print(sample_imgs.shape)
#                 sample_imgs = torch.cat([real, sample_imgs], dim = 0)
#                 grid = torchvision.utils.make_grid(sample_imgs)
#                 self.logger.experiment.add_image('generated_images', grid, self.global_step)
        
#         for _ in range(self.disc_freq):
# #         # train discriminator
# #         if optimizer_idx == 0:
#             noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
#             fake = self.gen(condition, noise)
#             disc_real = self.disc(condition, real).reshape(-1)
#             disc_fake = self.disc(condition, fake).reshape(-1)
#             gp = self.gradient_penalty(condition, real, fake)
#             loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.lambda_gp*gp
# #             self.logger.experiment.add_scalar('discriminator_loss', loss_disc, self.current_epoch)
#             self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
# #             return loss_disc
#             disc_opt.zero_grad()
# #             self.disc.zero_grad()
#             self.manual_backward(loss_disc, retain_graph=True)
# #             loss_disc.backward(retain_graph=True)
#             disc_opt.step()
        
# #         #train generator
# #         elif optimizer_idx ==1:
# #         noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
# #         fake = self.gen(condition, noise)
#         gen_fake = self.disc(condition, fake).reshape(-1)
#         loss_gen = -torch.mean(gen_fake)
#         self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
# #         return loss_gen 
#         gen_opt.zero_grad()
# #         self.gen.zero_grad()
#         self.manual_backward(loss_gen)
# #         loss_gen.backward(retain_graph=True)
#         gen_opt.step()
        

        
    
# #     def training_epoch_end(self, outputs):
# #         noise = torch.randn(self.num_to_visualise, *self.noise_shape, device = self.device)

# #         # log sampled images
# #         sample_imgs = self(torch.randint(low=0, high = self.num_classes - 1, size=(self.num_to_visualise,), device=self.device), noise)
# #         grid = torchvision.utils.make_grid(sample_imgs)
# #         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        
#     def configure_optimizers(self):
#         gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2))
#         disc_opt = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
# #         return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
#         return gen_opt, disc_opt
        
    
class LeinGANGP(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, channels_img, img_size, embed_size, 
                      b1 = 0.0, b2 = 0.9, lr = 1e-4, lambda_gp = 10, cond_idx = 0, real_idx = 1, disc_spectral_norm = False): # fill in
        super().__init__()
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.disc_freq, self.gen_freq = 5, 1
        self.noise_shape = noise_shape
        self.lambda_gp = lambda_gp
        self.gen = generator()
        self.disc = discriminator()
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        if disc_spectral_norm:
            self.disc.apply(self.add_sn)
        
    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            return nn.utils.parametrizations.spectral_norm(m)
        else:
            return m
        
    def forward(self, condition, noise):
        return self.gen(condition, noise)
    
    def gradient_penalty(self, condition, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1,C,H,W)
        interpolated_images = real*epsilon + fake*(1-epsilon)
        interpolated_images.requires_grad = True
        mixed_scores = self.disc(condition, interpolated_images)
        
        gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores, 
                    grad_outputs = torch.ones_like(mixed_scores), 
                    create_graph=True, 
                    retain_graph = True)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1)**2)
        return gradient_penalty
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step%100==0:
            with torch.no_grad():
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
        #         # log sampled images
                sample_imgs = self.gen(condition, noise)
                sample_imgs = torch.cat([real, sample_imgs], dim = 0)
#                 print(sample_imgs.shape)
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, self.global_step)
                grid = torchvision.utils.make_grid(condition)
                self.logger.experiment.add_image('input_images', grid, self.global_step)
                
        
#         # train discriminator
        if optimizer_idx == 0:
            noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            fake = self.gen(condition, noise)
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake).reshape(-1)
            gp = self.gradient_penalty(condition, real, fake)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.lambda_gp*gp
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc

        
#         #train generator
        elif optimizer_idx ==1:
            noise = torch.randn(real.shape[0], *self.noise_shape, device = self.device)
            fake = self.gen(condition, noise)
            gen_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen 
        
        
    def configure_optimizers(self):
        gen_opt = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        disc_opt = optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
#         return gen_opt, disc_opt