from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torchvision
from torchvision.utils import save_image

from networks.base.base_global_local_discriminating import BaseGlobalLocalDiscriminating
from networks.base.base_vae import BaseVAE
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder
from networks.intro_vae import Decoder
from utils import pytorch_util as ptu


class SSuperGlobalLocalDiscriminating(BaseGlobalLocalDiscriminating):
    def __init__(self,
                 generator: nn.Module,
                 output_img_size,
                 panel_img_size):
        super().__init__(generator)
        self.output_img_size = output_img_size
        self.panel_img_size = panel_img_size

    def forward(self, **kwargs) -> List[Tensor]:
        return self.generator(**kwargs)

    def create_local_discriminator(self) -> nn.Module:
        pass

    def create_global_discriminator(self) -> nn.Module:
        pass

    def sample(self, size: int) -> Tensor:
        return self.generator.sample(size)

    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)
