import torch
import torch.nn as nn
import numpy as np

class DCGANGenerator(nn.Module):
    
    def __init__(self, image_size, nc, nz, ngf):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
            )
        self.model.apply(self.weights_init)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # Normal Distribution with 0 mean 0.02 std
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.model(x)
