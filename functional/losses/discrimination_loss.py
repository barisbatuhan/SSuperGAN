import torch.nn as nn
import torch
import torch.nn.functional as F


def discrimination_loss(real_output, fake_output):
    return - 0.5 * real_output.log().mean() - 0.5 * (1 - fake_output).log().mean()