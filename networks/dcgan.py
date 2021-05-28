import argparse
import math
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
from utils import pytorch_util as ptu

from networks.base.base_gan import BaseGAN
from data.datasets.golden_faces import *

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class DCGAN(BaseGAN):
    def __init__(self, ngpu, image_size, nc, nz, ngf, ndf):
        super().__init__()
        self.ngpu = ngpu
        self.image_size = image_size
        self.nc = nc  # Number of channels of input
        self.nz = nz  # Latent dimension size
        self.ngf = ngf  # Size of feature maps in generator
        self.ndf = ndf  # Size of feature maps in discriminator
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.criterion = nn.BCELoss()

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.generator.apply(self.weights_init)

    def bce_loss(self, output, label):
        return self.criterion(output, label)

    def create_generator(self):
        generator = nn.Sequential(
            nn.ConvTranspose2d(self.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16, 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*8) x 4 x4
            nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid())

        return discriminator

    def create_generic_discriminator(self, im_size):
        if not math.log(im_size, 2).is_integer():
            raise AssertionError("im_size should be power of 2")
        intermediate_layer_size = int(math.log(im_size, 2)) - 3
        modules = [nn.Conv2d(self.nc, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.LeakyReLU(0.2, inplace=True)]
        for i in range(0, intermediate_layer_size):
            ndf_coeff = 2 ** (i + 1)
            modules.extend([nn.Conv2d(self.ndf * ndf_coeff // 2, self.ndf * ndf_coeff, kernel_size=4, stride=2,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(self.ndf * ndf_coeff),
                            nn.LeakyReLU(0.2, inplace=True)])
        modules.extend(
            [nn.Conv2d(self.ndf * (2 ** intermediate_layer_size), 1, kernel_size=4, stride=1, padding=0, bias=False),
             nn.Sigmoid()])
        discriminator = nn.Sequential(*modules)
        return discriminator

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # Normal Distribution with 0 mean 0.02 std
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def sample(self, size=10):
        with torch.no_grad():
            # Check how the generator is doing by saving G's output on fixed_noise
            noise = torch.randn(size, self.nz, 1, 1, device=ptu.device)
            fake_gen = self.generator(noise).detach().cpu()
            # -0.5 0.5
        # fake = (fake/2) + 0.5

        return fake_gen


if __name__ == "__main__":
    pass
    # TODO: define nc, nz, ngfi ndf
    #model = DCGAN(1, 64, nc, nz, ngf, ndf)
    #print(model.generator)
