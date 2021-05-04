import torch.nn as nn
import torch
import torch.nn.functional as F


def reconstruction_loss(x, x_recon, log_scale):
    """
    Retrieved from:
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    """
    dist = torch.distributions.Normal(x_recon, torch.exp(log_scale))
    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))
    
    
