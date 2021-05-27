import torch.nn as nn
from networks.contextual_networks.shared import DisConvModule
from typing import List


class GlobalDis(nn.Module):
    def __init__(self, c_num, input_dim=3, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = input_dim
        self.cnum = c_num
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        # below is from the original impl
        self.linear = nn.Linear(self.cnum * 4 * 16 * 16, 1)
        # below was modification for 128 * 128 panel size
        # self.linear = nn.Linear(self.cnum * 16 * 16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x