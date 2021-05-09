import torch
import torch.nn.functional as F

def kl_loss(z, mu, std):

    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_q = q.log_prob(z)
    log_p = p.log_prob(z)

    return F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
