import torch.nn as nn
import torch
import torch.nn.functional as F


def kl_loss(z, mu, std):
    """
    Retrieved from:
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    """
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    return (log_qzx - log_pz).sum(-1).mean()
