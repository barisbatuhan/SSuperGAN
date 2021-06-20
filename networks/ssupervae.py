import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torchvision

# Models
from networks.base.base_vae import BaseVAE
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder
from networks.intro_vae import Decoder

# Helpers
from utils import pytorch_util as ptu


class SSuperVAE(BaseVAE):

    def __init__(self,
                 # common parameters
                 backbone,
                 latent_dim=256,
                 embed_dim=256,
                 use_lstm=False,
                 # plain encoder parameters
                 seq_size=3,
                 # VAE parameters
                 decoder_channels=[64, 128, 256, 512],
                 gen_img_size=64,
                 # lstm encoder parameters
                 lstm_bidirectional=False,
                 lstm_hidden=256,
                 lstm_dropout=0,
                 fc_hidden_dims=[],
                 fc_dropout=0,
                 num_lstm_layers=1,
                 masked_first=True):
        super(SSuperVAE, self).__init__()

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
                                                 lstm_bidirectional=lstm_bidirectional,
                                                 fc_hidden_dims=fc_hidden_dims,
                                                 fc_dropout=fc_dropout,
                                                 num_lstm_layers=num_lstm_layers,
                                                 masked_first=masked_first)

        self.decoder = Decoder(
            hdim=latent_dim, channels=decoder_channels, image_size=gen_img_size)

        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )

    def forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        x_recon = self.decode(z)
        return z, None, mu, x_recon, lg_std

    def encode(self, x):
        return self.encoder(x)

    def generate(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def sample(self, size: int):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        return self.decode(z)

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)
