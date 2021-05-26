from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from networks.base.base_vae import BaseVAE
from networks.generic_vae import GenericVAE
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
from utils.image_utils import *


class VAETrainer(BaseTrainer):
    def __init__(self,
                 model: BaseVAE,
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
                 save_dir=base_dir + 'playground/vae/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         {"optimizer": optimizer},
                         {"scheduler": scheduler},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        # TODO: becareful about best loss here this might override the actual best loss
        #  in case of continuation of training
        best_loss = 99999999

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_vae(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_loss(self.test_loader)
            else:
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0}

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

            if self.checkpoint_every_epoch:
                self.save_checkpoint(current_loss={
                    "train_losses": train_losses,
                    "test_losses": test_losses
                },
                    current_epoch=epoch)
            
            metric_recorder.update_metrics(train_losses, test_losses)
            metric_recorder.save_recorder()
        return train_losses, test_losses

    @torch.no_grad()
    def eval_loss(self, data_loader):
        self.model.eval()
        total_losses = OrderedDict()
        for batch in data_loader:

            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].to(ptu.device), batch[1].to(ptu.device)
            else:
                x, y = batch.to(ptu.device), None

            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test --> '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f'{k} {total_losses[k]:.4f}'
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

            # This is buggy because when mask is included
            # list length is 3
            # TODO: fix this
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].to(ptu.device), batch[1].to(ptu.device)
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device),  batch[2].to(ptu.device)
            else:
                x, y = batch.to(ptu.device), None

            self.optimizer.zero_grad()
            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y

            out = self.criterion(z, target, mu_z, mu_x, logstd_z)
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
                pbar.update(x.shape[0])

        self.scheduler.step()
        self.model.save_samples(10, self.save_dir + '/results/' + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses
