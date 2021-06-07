import torch
import torch.nn as nn

class DCGANDiscriminator(nn.Module):
    
    def __init__(self, image_size, nc, nz, ndf):
        super().__init__()
        
        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
                
        img_len = image_size if type(image_size) == int else min(image_size)
        img_len = img_len // 2
        max_rng = 8 * ndf
        
        while img_len > 4:
            layers.append(nn.Conv2d(min(max_rng, ndf), min(max_rng, 2*ndf), kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(min(max_rng, 2*ndf)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ndf *= 2
            img_len = img_len // 2
                
        layers.append(nn.Conv2d(min(max_rng, ndf), 1, kernel_size=img_len, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)