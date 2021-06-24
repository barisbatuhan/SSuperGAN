import torch
import torch.nn as nn

class DCGANDiscriminator(nn.Module):
    
    def __init__(self, image_size, nc, nz, ndf, normalize="layer"):
        super().__init__()
        
        if normalize == "batch":
            norm = nn.BatchNorm2d
        elif normalize == "instance":
            norm = nn.InstanceNorm2d
        elif normalize == "layer":
            norm = nn.LayerNorm
        else:
            raise NotImplementedError
        
        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
                
        img_len = image_size if type(image_size) == int else min(image_size)
        img_len = img_len // 2
        max_rng = 8 * ndf
        
        while img_len > 4:
            
            IN, OUT, H, W = min(max_rng, ndf), min(max_rng, 2*ndf), img_len // 2, img_len // 2
            
            layers.append(nn.Conv2d(IN, OUT, kernel_size=4, stride=2, padding=1, bias=False))
            norm_in = OUT if normalize != "layer" else [OUT, H, W]
            layers.append(norm(norm_in))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            ndf, img_len = ndf * 2, img_len // 2 # update parameters
                
        layers.append(nn.Conv2d(OUT, 1, kernel_size=img_len, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)