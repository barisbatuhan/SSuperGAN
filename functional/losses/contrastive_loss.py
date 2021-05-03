import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from utils import pytorch_util as ptu
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Part 2.1
    The exact loss function is L(W, Y,~X1,~X2) = (1−Y)1/2(DW)^2 + (Y)1/2{max(0, m−DW)}^2
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, batch, acc_threshold=0.5):
        output1 = batch[0]
        output2 = batch[1]
        label = batch[2]
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        pred = ptu.get_numpy(euclidean_distance).ravel() < acc_threshold
        label = (ptu.get_numpy(label) == 0).ravel()
        acc = sum(pred == label) / len(euclidean_distance)

        return OrderedDict(loss=loss_contrastive, acc=acc)
