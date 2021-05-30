import enum

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models import *
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F

from networks.base.base_gan import BaseGAN
from utils import pytorch_util as ptu
from networks.mlp import MLP
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# Alternative Implementation: https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
# DUL Course Implementation:https://github.com/rll/deepul/blob/master/homeworks/solutions/hw4_solutions.ipynb

class BiGAN(BaseGAN):
    def __init__(self,
                 image_dim,
                 image_channel=3,
                 latent_dim=100,
                 g_hidden_size=1024,
                 d_hidden_size=1024,
                 e_hidden_size=1024,
                 ):
        super(BiGAN, self).__init__()
        self.image_channel = image_channel
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.g_hidden_size = g_hidden_size
        self.d_hidden_size = d_hidden_size
        self.e_hidden_size = e_hidden_size
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.encoder = self.create_encoder()
        self.latent_dist = Normal(ptu.FloatTensor([0.0],
                                                  torch_device=ptu.device),
                                  ptu.FloatTensor([1.0],
                                                  torch_device=ptu.device))

    def create_encoder(self, **kwargs) -> nn.Module:
        image_dim = self.image_dim
        hidden_size = self.e_hidden_size
        latent_dim = self.latent_dim
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_channel * (image_dim ** 2), hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_dim),
        )
        return encoder

    def create_generator(self, **kwargs) -> nn.Module:
        g_input_dim = self.latent_dim
        g_hidden_size = self.g_hidden_size
        output_dim = self.image_dim
        generator = nn.Sequential(
            nn.Linear(g_input_dim, g_hidden_size),
            nn.ReLU(),
            nn.Linear(g_hidden_size, g_hidden_size),
            nn.BatchNorm1d(g_hidden_size, affine=False),
            nn.ReLU(),
            nn.Linear(g_hidden_size, self.image_channel * (output_dim ** 2)),
            nn.Tanh(),
            Reshape(-1, self.image_channel, output_dim, output_dim)
        )
        return generator

    def create_discriminator(self, **kwargs) -> nn.Module:
        d_input_dim = self.image_channel * (self.image_dim ** 2) + self.latent_dim
        d_hidden_size = self.d_hidden_size
        discriminator = nn.Sequential(
            nn.Linear(d_input_dim, d_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_size, d_hidden_size),
            nn.BatchNorm1d(d_hidden_size, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_size, 1),
            nn.Sigmoid()
        )
        return discriminator
    
    # Next 3 functions will be the same for all the GAN networks implemented
    # -----------------------------------------------------------------------
    
    def generate(self, latents: Tensor) -> Tensor:
        return self.generator(latents)
    
    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)
    
    def discriminate(self, image: Tensor, latent: Tensor) -> Tensor:
        batchsize = image.shape[0]
        input_tensor = torch.cat((latent, image.view(batchsize, -1)), dim=1)
        return self.discriminator(input_tensor)
    
    # -----------------------------------------------------------------------
    
    def sample_latent(self, size: int) -> Tensor:
        return self.latent_dist \
            .sample(sample_shape=(size, self.latent_dim)) \
            .squeeze()

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        self.generator.eval()
        self.encoder.eval()
        x = x.view(-1, self.image_channel, self.image_dim, self.image_dim)
        z = self.encoder.forward(x)
        reconstruction = self.generator.forward(z)
        return reconstruction

    @torch.no_grad()
    def sample(self, size: int) -> Tensor:
        self.generator.eval()
        latents = self.sample_latent(size)
        return self.generate(latents)

    # TODO: This might be improved
    @torch.no_grad()
    def interpolate(self, batchsize: int) -> Tensor:
        intervals = torch.linspace(-1, 1, batchsize).cuda()
        latents = torch.repeat_interleave(intervals, self.latent_dim)
        return self.generate(latents.view(batchsize, -1))
