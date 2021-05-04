import enum

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

from networks.bigan import BiGAN
from utils import pytorch_util as ptu
import numpy as np


# TODO: implement Wasserstein Loss
class BidirectionalDiscriminatorLossType(enum.Enum):
    VANILLA = 1
    # This can be used as a source: https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
    WASSERSTEIN = 2


class BidirectionalDiscriminatorLoss(nn.Module):
    def __init__(self, loss_type: BidirectionalDiscriminatorLossType):
        super(BidirectionalDiscriminatorLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self,
                bigan: BiGAN,
                batch):
        batchsize = batch.shape[0]
        if self.loss_type == BidirectionalDiscriminatorLossType.VANILLA:
            z_fake = bigan.sample_latent(batchsize)
            z_real = bigan.encoder.forward(batch)

            x_fake = bigan.generator.forward(z_fake)
            x_real = batch

            fake_input = torch.cat((z_fake, x_fake.view(batchsize, -1)), dim=1)
            real_input = torch.cat((z_real, x_real.view(batchsize, -1)), dim=1)

            disc_fake_output = bigan.discriminator.forward(fake_input)
            disc_real_output = bigan.discriminator.forward(real_input)

            d_loss = - 0.5 * disc_real_output.log().mean() - 0.5 * (1 - disc_fake_output).log().mean()
            return OrderedDict(loss=d_loss)
        elif self.loss_type == BidirectionalDiscriminatorLossType.WASSERSTEIN:
            raise NotImplementedError
        else:
            raise NotImplementedError
