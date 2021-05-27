import torch.nn as nn
from networks.contextual_networks.shared import DisConvModule
from typing import List


# TODO: create controlled local dis
# TODO: complete ssuper global local
# TODO: do it with wgan
# TODO: create playground
# TODO: create trainer
# TODO: train a plain model
class BaseDiscriminator(nn.Module):
    def __init__(self,
                 spatial_dims: List[int],
                 intermediate_channel_num,
                 input_dim=3):
        super(BaseDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.spatial_dims = spatial_dims
        self.intermediate_channel_num = intermediate_channel_num
        self.dis_conv_module = DisConvModule(self.input_dim, self.intermediate_channel_num)

        spatial_output_size_coefficient = 1
        for dim in spatial_dims:
            spatial_output_size_coefficient *= dim // 16

        self.linear = nn.Linear(self.intermediate_channel_num * 4 * spatial_output_size_coefficient, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
