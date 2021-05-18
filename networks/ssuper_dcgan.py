import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torchvision

# Models
from networks.base.base_vae import BaseVAE
from networks.base.base_gan import BaseGAN
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder
from networks.intro_vae import Decoder
from networks.dcgan import DCGAN

# Losses
from functional.losses.kl_loss import kl_loss
from functional.losses.reconstruction_loss import reconstruction_loss

# Helpers
from utils import pytorch_util as ptu
from utils.config_utils import read_config,Config

class SSuperDCGAN(nn.Module):
    
    def __init__(self, 
                 # common parameters
                 backbone, 
                 latent_dim=256, 
                 embed_dim=256,
                 use_lstm=False,
                 # plain encoder parameters
                 seq_size=3,
                 gen_img_size=64,
                 # lstm encoder parameters
                 lstm_hidden=256,
                 lstm_dropout=0,
                 fc_hidden_dims=[],
                 fc_dropout=0,
                 num_lstm_layers=1,
                 masked_first=True):
        super(SSuperDCGAN, self).__init__()
        
        self.latent_dim = latent_dim
        
        if not use_lstm:
            self.encoder = PlainSequentialEncoder(backbone, 
                                                  latent_dim=latent_dim, 
                                                  embed_dim=embed_dim, 
                                                  seq_size=seq_size)
        else:
            self.encoder = LSTMSequentialEncoder(backbone,
                                                 latent_dim=latent_dim,
                                                 embed_dim=embed_dim,
                                                 lstm_hidden=lstm_hidden,
                                                 lstm_dropout=lstm_dropout,
                                                 fc_hidden_dims=fc_hidden_dims,
                                                 fc_dropout=fc_dropout,
                                                 num_lstm_layers=num_lstm_layers,
                                                 masked_first=masked_first)
        
        
        

       
        config = read_config(Config.DCGAN)

        nc = config.nc
        nz = config.nz
        ngf = config.ngf
        ndf = config.ndf
        ngpu = config.ngpu
        image_size = config.image_size

        self.dcgan = DCGAN(ngpu, image_size, nc, nz, ngf, ndf)


        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )
        
    def forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        x_recon = self.dcgan.generator(z)

        return z, None, mu, x_recon, lg_std
    
    def encode(self, x):
        return self.encoder(x)
    
    def generate(self, x):
        mu, _ = self.encode(x)
        return mu
    
    def decode(self, z):
        return self.dcgan.generator(z)
    
    def sample(self, size :int):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        return self.decode(z)
    
    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)