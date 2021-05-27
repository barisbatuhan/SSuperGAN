import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=64,
                 num_layers=2,
                 activate_function_type=nn.ReLU,
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func_type = activate_function_type
        self.num_layers = num_layers
        self.net = nn.ModuleList([])
        for i in range(num_layers):
            self.net.append(nn.Linear(in_features=input_size,
                                      out_features=hidden_size))
            input_size = hidden_size
            if self.activation_func_type is nn.LeakyReLU:
                slope = kwargs["slope"]
                self.net.append(nn.LeakyReLU(slope))
            else:
                self.net.append(self.activation_func_type())
        self.net.append(nn.Linear(in_features=hidden_size,
                                  out_features=output_size))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(len(x), -1)
        out = torch.flatten(x, start_dim=1, end_dim=-1)
        for layer in self.net:
            out = layer(out.float())
        return out
