import torch
import torch.nn as nn
import torchvision

from networks.panel_encoder.cnn_embedder import CNNEmbedder

class PlainSequentialEncoder(nn.Module):
    
    def __init__(self, backbone, latent_dim=256, embed_dim=256, seq_size=3):
        super(PlainSequentialEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        # Embeds the sequential image frames
        self.embedder = CNNEmbedder(backbone, embed_dim=embed_dim)
        # Used for computing mean and log std
        self.fc_mean = nn.Linear(seq_size*embed_dim, latent_dim)
        self.fc_lgstd = nn.Linear(seq_size*embed_dim, latent_dim)
    
    def forward(self, x):
        B = x.shape[0]
        outs = self.embedder(x).reshape(B, -1) # concatanated seq. panel outs
        
        mu = self.fc_mean(outs)
        lg_std = self.fc_lgstd(outs)
        
        return mu, lg_std