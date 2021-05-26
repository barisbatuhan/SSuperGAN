import torch
from torch.distributions.normal import Normal

# Models

from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder
from networks.dcgan import DCGAN
# Helpers
from utils import pytorch_util as ptu

import torch.nn as nn
from torchvision.utils import save_image


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
                 masked_first=True,
                 ngpu=1,
                 ngf=64,
                 ndf=64,
                 nc=3,
                 image_size=64):
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

        print(
            f"DCGAN PRAMS ngpu : {ngpu}  image_size : {image_size}  nc : {nc} latent_dim : {latent_dim}  ngf : {ngf} ndf {ndf}")
        self.dcgan = DCGAN(ngpu, image_size, nc, latent_dim, ngf, ndf)

        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )

    def forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        z = torch.unsqueeze(z, (2))
        z = torch.unsqueeze(z, (3))
        x_recon = self.dcgan.generator(z)

        return z, None, mu, x_recon, lg_std

    def encode(self, x):
        return self.encoder(x)

    def generate(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode(self, z):
        return self.dcgan.generator(z)

    def sample(self, size: int):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        z = torch.unsqueeze(z, (2))
        # print("Forward z shape ",z.shape)
        z = torch.unsqueeze(z, (3))
        # print("Sample z size : ",z.shape)
        return self.decode(z)

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)
