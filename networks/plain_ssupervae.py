import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torchvision

# Models
from networks.base.base_vae import BaseVAE
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.intro_vae import Decoder

# Losses
from functional.losses.kl_loss import kl_loss
from functional.losses.reconstruction_loss import reconstruction_loss

# Helpers
from utils import pytorch_util as ptu

class PlainSSuperVAE(BaseVAE):
    
    def __init__(self, 
                 backbone, 
                 latent_dim=256, 
                 embed_dim=256,
                 seq_size=3,
                 decoder_channels=[64, 128, 256, 512],
                 gen_img_size=64
                ):
        super(PlainSSuperVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = PlainSequentialEncoder(
            backbone, latent_dim=latent_dim, embed_dim=embed_dim, seq_size=seq_size)
        
        self.decoder = Decoder(
            hdim=latent_dim, channels=decoder_channels, image_size=gen_img_size)
        
        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )
        
    def forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        mu_x = self.decode(z)
        return z, None, mu, mu_x, lg_std
    
    def encode(self, x):
        return self.encoder(x)
    
    def generate(self, x):
        mu, _ = self.encode(x)
        return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def sample(self, size :int, current_device :int=0):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        return self.decode(z)
    
    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)