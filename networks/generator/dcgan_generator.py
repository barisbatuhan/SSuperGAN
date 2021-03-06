import torch
import torch.nn as nn
import numpy as np

class DCGANGenerator(nn.Module):
    
    def __init__(self, image_size, nc, nz, ngf, leaky=0, normalize="layer"):
        super().__init__()  
        
        if leaky == 0:
            activ = nn.ReLU()
        else:
            activ = nn.LeakyReLU(leaky, inplace=True)
        
        if normalize == "batch":
            norm = nn.BatchNorm2d
        elif normalize == "instance":
            norm = nn.InstanceNorm2d
        elif normalize == "layer":
            norm = nn.LayerNorm
        else:
            raise NotImplementedError
        
        layers = []
        mult_coeffs = {1024:8, 512:8, 216:8, 128:8, 64:8, 32:4, 16:2, 8:1}
        
        layers.append(nn.ConvTranspose2d(nz, 
                                         ngf * mult_coeffs[image_size], 
                                         kernel_size=4, stride=1, padding=0, bias=False)) 
        
        norm_in = [ngf*8, 4, 4] if normalize == "layer" else ngf*8
        layers.append(norm(norm_in))
        layers.append(activ)
        image_size, curr_size = image_size // 2, 8
        
        while image_size > 4:
            
            IN, OUT = ngf * mult_coeffs[image_size * 2], ngf * mult_coeffs[image_size] 
            H, W = curr_size, curr_size
            
            layers.append(nn.ConvTranspose2d(IN, OUT, kernel_size=4, stride=2, padding=1, bias=False))
            norm_in = [OUT, H, W] if normalize == "layer" else OUT
            layers.append(norm(norm_in))
            layers.append(activ)
            
            image_size, curr_size = image_size // 2, curr_size * 2
        
        layers.append(nn.ConvTranspose2d(OUT, nc, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
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
        if len(x.shape) < 4:
            for _ in range(len(x.shape), 4):
                x = x.unsqueeze(-1)
        return self.model(x)
