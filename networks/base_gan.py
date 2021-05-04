import torch.nn as nn
from torch import Tensor
from abc import abstractmethod
import torch
from torchvision.utils import save_image


class BaseGAN(nn.Module):
    def __init__(self) -> None:
        super(BaseGAN, self).__init__()

    def create_generator(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def create_discriminator(self, **kwargs) -> nn.Module:
        raise NotImplementedError
        
    def create_encoder(self, **kwargs) -> nn.Module:
        raise NotImplementedError
       
    def generate(self, latents: Tensor) -> Tensor:
        raise NotImplementedError
    
    def encode(self, image: Tensor) -> Tensor:
        raise NotImplementedError
    
    def discriminate(self, image: Tensor, latent: Tensor) -> Tensor:
        # if discriminator only uses image, just do not process latent or
        # or pass an empty latent vector
        raise NotImplementedError

    def sample_latent(self, batchsize: int) -> Tensor:
        raise NotImplementedError

    def sample(self, size: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, batchsize: int) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)
