import torch
import torch.nn as nn
from utils import pytorch_util as ptu
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor
from abc import abstractmethod
from torchvision.utils import save_image
from utils.image_utils import *

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, size: int) -> Tensor:
        raise RuntimeWarning()
        
    @torch.no_grad()
    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> Any:
        pass

    @abstractmethod
    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)


"""
    @abstractmethod
    def loss(self, inputs: Any, **kwargs) -> Any:
        pass
"""
