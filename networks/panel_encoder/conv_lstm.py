from torch import nn
import torch

"""
This implementation is taken from:
https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
"""

class ConvLSTM(nn.Module):
    def __init__(self, input_channel, filter_shape, kernel_size, stride=1, padding=1):
        super().__init__()
        
        num_filter, self._state_height, self._state_width = filter_shape
        
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs, states=None):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(inputs.get_device())
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(inputs.get_device())
        else:
            h, c = states

        outputs = []
        for index in range(inputs.shape[0]):
            x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)