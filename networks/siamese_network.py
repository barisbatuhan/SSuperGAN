import enum

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models import *
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
from utils import pytorch_util as ptu
from networks.mlp import MLP


class SiameseBackbone(enum.Enum):
    BASIC = 1
    RESNET_18 = 2


# TODO: Add backbone options
class SiameseNetwork(nn.Module):
    def __init__(self,
                 image_dim,
                 backbone=SiameseBackbone.RESNET_18):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.image_dim = image_dim

        fc_input_size = None
        if backbone is SiameseBackbone.BASIC:
            self.cnn1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, 4, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
            )
            fc_input_size = 8 * image_dim * image_dim
        elif backbone is SiameseBackbone.RESNET_18:
            self.cnn1 = nn.Sequential(*(list(resnet18().children())[0:8]))
            fc_input_size = 2048
        else:
            raise NotImplementedError

        self.fc1 = MLP(input_size=fc_input_size,
                       output_size=5,
                       hidden_size=500,
                       num_layers=2)

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, batch):
        input1 = batch[0]
        input2 = batch[1]
        labels = batch[2]
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2, labels
