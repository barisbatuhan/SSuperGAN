from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from functional.losses.bi_discriminator_loss import BidirectionalDiscriminatorLoss
from networks.base.base_gan import BaseGAN
from networks.bigan import BiGAN
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *
from typing import Tuple


class BiGANTrainer(object):
    def __init__(self,
                 model: BaseGAN,
                 criterion: BidirectionalDiscriminatorLoss,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer_discriminator,
                 optimizer_generator,
                 scheduler_disc=None,
                 scheduler_gen=None,
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 generator_update_round=1,
                 # Possible Value: (-0.01, 0.01)
                 disc_weight_clipping: Tuple[float, float] = None
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.quiet = quiet
        self.epochs = epochs
        self.best_loss_action = best_loss_action
        self.criterion = criterion
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_generator = optimizer_generator
        self.scheduler_disc = scheduler_disc
        self.scheduler_gen = scheduler_gen
        self.grad_clip = grad_clip
        self.disc_weight_clipping = disc_weight_clipping
        self.generator_update_round = generator_update_round

    # TODO: if we were to add global & local discriminator dataloader needs to be updated
    # along with the model and basically two disc heads needs to be
    def train_bigan(self):
        self.model.train()
        metric_recorder = MetricRecorder(save_dir=base_dir + 'playground/bigan/results/')

        losses = OrderedDict()
        for epoch in range(self.epochs):
            logging.info("epoch start: " + str(epoch))
            self.model.train()
            if not self.quiet:
                pbar = tqdm(total=len(self.train_loader.dataset))
            for count, batch in enumerate(self.train_loader):
                batch = batch.cuda().float()

                # do a minibatch update
                self.optimizer_discriminator.zero_grad()
                d_out = self.criterion.forward(self.model, batch)
                d_out['loss'].backward()

                # Weight Clipping For Wesserstein Loss
                if self.disc_weight_clipping is not None:
                    for p in self.model.discriminator.parameters():
                        p.data.clamp_(self.disc_weight_clipping[0], self.disc_weight_clipping[0])

                d_out['discriminator_loss'] = d_out.pop('loss')
                self.optimizer_discriminator.step()

                g_out = {}
                # generator and encoder update
                if count % self.generator_update_round == 0:
                    self.optimizer_generator.zero_grad()
                    g_out = self.criterion.forward(self.model, batch)
                    g_out['loss'] = -1 * g_out['loss']
                    g_out['loss'].backward()
                    g_out['generator_loss'] = g_out.pop('loss')
                    self.optimizer_generator.step()

                desc = f'Epoch {epoch}'
                for out in [d_out, g_out]:
                    for k, v in out.items():
                        if k not in losses:
                            losses[k] = []
                        losses[k].append(v.item())
                        avg = np.mean(losses[k][-50:])
                        desc += f', {k} {avg:.4f}'

                if not self.quiet:
                    logging.info(desc)
                    pbar.set_description(desc)
                    if type(batch) is list:
                        pbar.update(batch[0].shape[0])
                    else:
                        pbar.update(batch.shape[0])

            if not self.quiet:
                pbar.close()
            # save training metrics
            metric_recorder.update_metrics(train=losses, test=None)
            metric_recorder.save_recorder()
            # step the learning rate
            self.scheduler_disc.step()
            self.scheduler_gen.step()
            # TODO: HOW DO WE DECIDE BEST LOSS ACTION?
            # MAYBE WE SHOULD CHECK RECONSTRUCTION LOSS ?
            # SO SAVE MODEL AFTER EACH EPOCH
            self.best_loss_action(self.model,
                                  str(losses['generator_loss'])
                                  + " "
                                  + str(losses['discriminator_loss']))
            # creating and saving images after each epoch
            self.model.save_samples(10, base_dir + 'playground/bigan/results/ + 'f'epoch{epoch}_samples.png')

        return losses
