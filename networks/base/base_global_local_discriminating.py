import torch.nn as nn
from torch import Tensor
import torch
from typing import Tuple, List


class BaseGlobalLocalDiscriminating(nn.Module):
    def __init__(self,
                 generator: nn.Module) -> None:
        super(BaseGlobalLocalDiscriminating, self).__init__()
        self.global_discriminator = self.create_global_discriminator()
        self.local_discriminator = self.create_local_discriminator()
        self.generator = generator

    def forward(self, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def dis_forward(self, is_local, ground_truth, generated) -> Tuple[Tensor, Tensor]:
        assert ground_truth.size() == generated.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, generated], dim=0)
        batch_output = self.local_discriminator(batch_data) if is_local else self.global_discriminator(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        return real_pred, fake_pred

    def create_local_discriminator(self) -> nn.Module:
        raise NotImplementedError

    def create_global_discriminator(self) -> nn.Module:
        raise NotImplementedError
