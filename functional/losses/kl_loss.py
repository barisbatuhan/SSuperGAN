import torch
import torch.nn.functional as F

"""
def kl_loss(z, mu, std):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_q = q.log_prob(z)
    log_p = p.log_prob(z)

    return F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
"""

# def kl_loss(z, mu, std):
#     p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
#     q = torch.distributions.Normal(mu, std)

#     kl_dvergence = torch.distributions.kl.kl_divergence(q,p)
#     kl_loss = kl_dvergence.sum(1).mean()
#     return kl_loss


def kl_loss(mu, logvar, prior_mu=0):
    v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    v_kl = v_kl.sum(dim=-1).mul_(-0.5)
    return v_kl.mean()

 
