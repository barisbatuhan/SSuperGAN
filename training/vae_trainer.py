from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from networks.generic_vae import GenericVAE
from utils.structs.metric_recorder import *
from utils.logging_utils import *


class VAETrainer(object):
    def __init__(self,
                 model: GenericVAE,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer,
                 scheduler=None,
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.quiet = quiet
        self.epochs = epochs
        self.best_loss_action = best_loss_action
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip

    def train_epochs(self):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=base_dir + 'playground/vae/results/')
        epochs = self.epochs
        best_loss = 99

        train_losses, test_losses = OrderedDict(), OrderedDict()
        for epoch in range(epochs):
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_vae(epoch)
            test_loss = self.eval_loss(self.test_loader)

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss[k])
                if k == "loss":
                    current_test_loss = test_loss[k]
                    if current_test_loss < best_loss:
                        best_loss = current_test_loss
                        if self.best_loss_action != None:
                            self.best_loss_action(self.model, best_loss)

            metric_recorder.update_metrics(train_losses, test_losses)
            metric_recorder.save_recorder()
        return train_losses, test_losses

    @torch.no_grad()
    def eval_loss(self, data_loader):
        self.model.eval()
        total_losses = OrderedDict()
        for batch in data_loader:
            out = self.model(batch)
            out = self.criterion(*out)
            for k, v in out.items():
                if type(batch) is list:
                    total_losses[k] = total_losses.get(k, 0) + v.item() * batch[0].shape[0]
                else:
                    total_losses[k] = total_losses.get(k, 0) + v.item() * batch.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not self.quiet:
            print(desc)
            logging.info(desc)
        return total_losses

    def train_vae(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        losses = OrderedDict()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            out = self.model(batch)
            # z, inputs, mu_z, mu_x, logstd_z
            out = self.criterion(*out)
            out['loss'].backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            desc = f'Epoch {epoch}'
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v.item())
                avg = np.mean(losses[k][-50:])
                desc += f', {k} {avg:.4f}'

            if not self.quiet:
                pbar.set_description(desc)
                if type(batch) is list:
                    pbar.update(batch[0].shape[0])
                else:
                    pbar.update(batch.shape[0])
        self.scheduler.step()
        self.model.save_samples(10, base_dir + 'playground/vae/results/ + 'f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses
