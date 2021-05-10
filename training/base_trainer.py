from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
from typing import Dict

from networks.base.base_vae import BaseVAE
from networks.generic_vae import GenericVAE
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu


class BaseTrainer(object):
    def __init__(self,
                 model,
                 model_name,
                 criterion,
                 epochs,
                 save_dir,
                 optimizers: Dict[str, optim.Optimizer],
                 schedulers: Dict[str, object] = {},
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 checkpoint_every_epoch=False
                 ):
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.model_name = model_name
        self.quiet = quiet
        self.epochs = epochs
        self.criterion = criterion
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        self.best_loss_action = best_loss_action
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_path = self.save_dir + 'checkpoints/'

    def save_checkpoint(self, current_epoch, current_loss):
        """
        IMPORTANT: /checkpoints folder should be avaialbe under save_dir
        Saves checkpoint at self.checkpoint_path + self.model_name + "checkpoint" + ".pth"
        location
        :param current_epoch:
        :param current_loss:
        :return:
        """
        checkpoint_path = self.checkpoint_path + self.model_name + "-checkpoint" + ".pth"
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': current_loss,
        }

        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        for key, value in self.schedulers.items():
            checkpoint[key] = value.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, alternative_chkpt_path=None):
        """
        Loads model completely from a saved checkpoint
        :param alternative_path: path to be loaded if provided
        :return: epoch, loss
        """
        if alternative_chkpt_path is None:
            checkpoint_path = self.checkpoint_path + self.model_name + "-checkpoint" ".pth"
        else:
            checkpoint_path = alternative_chkpt_path
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        for key, value in self.optimizers.items():
            value.load_state_dict(checkpoint[key])

        for key, value in self.schedulers.items():
            value.load_state_dict(checkpoint[key])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return epoch, loss
