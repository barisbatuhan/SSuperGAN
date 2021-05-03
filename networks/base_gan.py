import torch.nn as nn
from torch import Tensor
from abc import abstractmethod


class BaseGAN(nn.Module):
    def __init__(self) -> None:
        super(BaseGAN, self).__init__()

    def create_generator(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def create_discriminator(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def sample_latent(self, batchsize: int) -> Tensor:
        raise NotImplementedError

    def sample_fake_with_latent(self, latents: Tensor) -> Tensor:
        raise NotImplementedError

    def random_sample_generator(self, batchsize: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def interpolate_sample_generator(self, batchsize: int) -> Tensor:
        pass
