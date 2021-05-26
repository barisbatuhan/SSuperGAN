import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import pytorch_util as ptu


def reconstruction_loss_distributional(x, x_recon, log_scale=None):
    """
    :param x: reconstructed
    :param x_recon: mean of distribution
    :param log_scale: if None 1 will be used, if not it is log scale std of distribution
    :return: distributional reconstruction loss
    Retrieved from:
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
   """
    if log_scale is None:
        dist = torch.distributions.Normal(x_recon, ptu.FloatTensor([1.0],
                                                                   torch_device=ptu.device))
    else:
        dist = torch.distributions.Normal(x_recon, torch.exp(log_scale))
    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3)).mean()


def l1_loss(x, x_recon):
    return torch.abs(x - x_recon).mean()