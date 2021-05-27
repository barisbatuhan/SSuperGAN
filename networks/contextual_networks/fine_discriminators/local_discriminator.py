from typing import List

import torch.nn as nn
from networks.contextual_networks.shared import gen_conv, dis_conv, DisConvModule


class LocalDis(nn.Module):
    def __init__(self, c_num, input_dim=3):
        super(LocalDis, self).__init__()
        self.input_dim = input_dim
        self.cnum = c_num

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum * 8 * 8, 1)
        # Below is from the original implementation
        # self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x