import torch
import torch.nn as nn
import torchvision

import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# a backbone example: 'efficientnet-b5'
# for efficientnet model: https://github.com/lukemelas/EfficientNet-PyTorch

class CNNEmbedder(nn.Module):
    
    def __init__(self, backbone, use_linear=True, embed_dim=256):
        super(CNNEmbedder, self).__init__()
        
        self.embed_dim = embed_dim
        self.use_linear = use_linear
        self.backbone = backbone
        
        if not use_linear:
            embed_dim = 1 # to reduce the memory size
 
        if backbone == "resnet50":
            self.model = torch.nn.Sequential(
                *(list(models.resnet50(pretrained=True).children())[:-1]),
                nn.Linear(2048, embed_dim)
            )
        
        elif "efficientnet" in backbone:
            self.model = EfficientNet.from_pretrained(backbone, num_classes=embed_dim)
        
        
    def forward(self, x):
        B, S, C, H, W = x.shape
        # Retrieved the embeddings for each of the panels
        if self.use_linear:
            outs = self.model(x.reshape(-1, C, H, W))
            return outs.reshape(B, S, -1)
        
        else:
            if "efficientnet" in self.backbone:
                outs = self.model.extract_features(x.reshape(-1, C, H, W))
            else:
                outs = self.model[:-2](x.reshape(-1, C, H, W))
            
            return outs.reshape(B, S, *outs.shape[1:])