import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from base.base_gan import BaseGAN

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1




class DCGAN(BaseGAN):
    def __init__(self, ngpu, image_size, nc, nz, ngf, ndf):
        super().__init__()
        self.ngpu = ngpu
        self.image_size = image_size
        self.nc  = nc  # Number of channels of input
        self.nz = nz   # Latent dimension size
        self.ngf = ngf # Size of feature maps in generator
        self.ndf = ndf # Size of feature maps in discriminator
        self.generator = self.create_generator()
        self.disciminator = self.create_discriminator()

    def loss(self):
        pass

    def create_generator(self):
        generator = nn.Sequential(
            nn.ConvTranspose2d(nz, out_channels=self.ngf*8, kernel_size = 4, stride = 1, padding =0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size =4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            )
        return generator


    def create_discriminator(self):
        discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16, 16
            nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*8) x 4 x4
            nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=1, padding=0,bias=False),
            nn.Sigmoid())

        return discriminator

        



    
        


if __name__== "__main__":
    model = DCGAN(1, 64, nc, nz, ngf, ndf)
    print(model.generator)



