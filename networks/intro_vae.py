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
from networks.generic_vae import GenericVAE
from utils import pytorch_util as ptu
from networks.mlp import MLP
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class Encoder(nn.Module):
    def __init__(self,
                 cdim=3,
                 hdim=512,
                 channels=[64, 128, 256, 512, 512, 512],
                 image_size=256):
        super(Encoder, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.fc = nn.Linear((cc) * 4 * 4, 2 * hdim)

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        return y
        # mu, logvar = y.chunk(2, dim=1)
        # return mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                 cdim=3,
                 hdim=512,
                 channels=[64, 128, 256, 512, 512, 512],
                 image_size=256):
        super(Decoder, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(hdim, cc * 4 * 4),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y


class IntroVAE(GenericVAE):
    """
    This is only the model, however, the loss function is extremely important for
    this paper. Maybe we can make use of that as well!
Source: https://github.com/hhb072/IntroVAE/blob/master/networks.py
Source PAPER: IntroVAE: Introspective Variational Autoencoders forPhotographic Image Synthesis
https://arxiv.org/pdf/1807.06358.pdf
    """

    def __init__(self,
                 cdim=3,
                 hdim=512,
                 channels=[64, 128, 256, 512, 512, 512],
                 image_size=256):
        """
        :param cdim: channel dimension
        :param hdim: latent (z) dimension
        :param channels: following needs to be considered => assert (2 ** len(channels)) * 4 == image_size
        :param image_size: input - output image size
        """
        encoder = Encoder(cdim, hdim, channels, image_size)
        decoder = Decoder(cdim, hdim, channels, image_size)
        super(IntroVAE, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       latent_dim=hdim)


if __name__ == '__main__':
    im_size = 64
    intro_vae = IntroVAE(image_size=im_size, channels=[64, 128, 256, 512])
    batch_like = ptu.randn(1, 3, im_size, im_size, torch_device=ptu.device)
    encoder_result = intro_vae.forward(batch_like)
    print(encoder_result)
