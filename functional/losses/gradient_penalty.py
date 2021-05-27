import torch
from utils import pytorch_util as ptu
import torch.nn as nn


def calculate_gradient_penalty(net: nn.Module,
                               real_data: torch.Tensor,
                               fake_data: torch.Tensor):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if ptu.gpu_enabled():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_().clone()

    disc_interpolates = net(interpolates)
    grad_outputs = torch.ones(disc_interpolates.size())

    if ptu.gpu_enabled():
        grad_outputs = grad_outputs.cuda()

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=grad_outputs, create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
