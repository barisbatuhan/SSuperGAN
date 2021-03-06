import torch.nn as nn
import torch
import torch.nn.functional as F

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
        dist = torch.distributions.Normal(x_recon, torch.FloatTensor([1.0]).cuda())
    else:
        dist = torch.distributions.Normal(x_recon, torch.exp(log_scale))
    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3)).mean()


def reconstruction_loss(x, x_recon, size_average=False):        
    bs = x.shape[0]
    reduction = "sum" if not size_average else "mean"
    return F.mse_loss(x_recon, x, reduction=reduction).div(bs)


def l1_loss(x, x_recon):
    return torch.abs(x - x_recon).mean()
