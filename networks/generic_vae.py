import torch
import torch.nn as nn

from networks.base.base_vae import BaseVAE
from utils import pytorch_util as ptu
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor
from torch.distributions.normal import Normal


class GenericVAE(BaseVAE):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 latent_dim=16,
                 kld_loss_weight=1):
        super(GenericVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kld_loss_weight = kld_loss_weight
        self.latent_dim = latent_dim
        self.latent_dist = Normal(ptu.FloatTensor([0.0],
                                                  torch_device=ptu.device),
                                  ptu.FloatTensor([1.0],
                                                  torch_device=ptu.device))

    """
    ON .rsample():
    The other way to implement these stochastic/policy gradients would be to use the reparameterization trick from the rsample() method, 
    where the parameterized random variable can be constructed via a parameterized deterministic function of a parameter-free random variable. 
    The reparameterized sample therefore becomes differentiable.
    """

    def encode(self, x):
        encoder_nn_output = self.encoder(x)
        mu, logstd = torch.chunk(encoder_nn_output, 2, dim=1)
        # sample z from q
        std = torch.exp(logstd)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        q.rsample()
        # q_phi or p_theta they are same
        return [z, mu, logstd]

    def decode(self, z):
        mu_x = self.decoder(z)
        return mu_x

    def forward(self, inputs: Tensor) -> Any:
        encoding_res = self.encode(inputs)
        z = encoding_res[0]
        mu_z = encoding_res[1]
        logstd_z = encoding_res[2]
        mu_x = self.decode(z)
        return z, inputs, mu_z, mu_x, logstd_z

    @torch.no_grad()
    def sample(self, size: int, **kwargs) -> Tensor:
        self.eval()
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze()
        samples = self.decode(z)
        return samples

    @torch.no_grad()
    def interpolations(self, base_images_as_numpy, steps=10):
        self.eval()
        double_n_row, c, h, w = base_images_as_numpy.shape
        n_row = double_n_row // 2
        x = ptu.from_numpy(base_images_as_numpy)
        encoding_res = self.encode(x)
        z = encoding_res[0]
        interpolation_line = torch.linspace(0, 1, steps=steps, device=ptu.device).view(-1, 1)
        starts, ends = torch.chunk(z, 2, dim=0)
        interpolations = ptu.zeros(0, n_row, 3, h, w)
        for line in interpolation_line:
            interpolation_column = starts * line + ends * (1 - line)
            interpolation_column = self.decode(interpolation_column)
            interpolations = torch.cat((interpolations, interpolation_column.view(1, -1, c, h, w)), dim=0)
        interpolations = torch.transpose(interpolations, 0, 1)
        interpolations = torch.reshape(interpolations, (-1, c, h, w))
        x = interpolations.permute(0, 2, 3, 1)
        x = ptu.get_numpy(x)
        return x

    """
        def loss(self, inputs: Any, **kwargs) -> Any:
            x = inputs
            z, x, mu_z, mu_x, logstd_z = self.forward(x)
            kl_loss = kl_divergence(z, mu_z, logstd_z)
            reconstruction_loss = -1 * reconstruction_likelihood(mu_x, x)
            loss = reconstruction_loss + self.kld_loss_weight * kl_loss
            return {VAELossType.ELBO: loss.mean(),
                    VAELossType.RECONSTRUCTION: reconstruction_loss.mean(),
                    VAELossType.KLD: kl_loss.mean()}
    """
