from collections import OrderedDict
import torch
from functional.losses.kl_loss import kl_loss
from functional.losses.reconstruction_loss import reconstruction_loss_distributional 
from functional.losses.reconstruction_loss import reconstruction_loss, l1_loss

def elbo(z,
         x,
         mu_z,
         mu_x,
         logstd_z,
         logstd_x=None,
         kl_loss_weight=1, 
         recon_loss_weight=1,
         l1_recon=False):
    
    kl_loss_val = kl_loss_weight * kl_loss(mu_z, logstd_z) 
    
    if l1_recon:
        recon_loss_val = recon_loss_weight * reconstruction_loss(x, mu_x, True) 
    else:
        recon_loss_val = -recon_loss_weight * reconstruction_loss_distributional(x, mu_x, logstd_x)
    
    loss = recon_loss_val + kl_loss_val
    
    return OrderedDict(loss=loss,
                       reconstruction_loss=recon_loss_val,
                       kl_loss=kl_loss_val)
