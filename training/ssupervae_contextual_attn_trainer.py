from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from networks.ssupervae_contextual_attentional import SSuperVAEContextualAttentional
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
import torchvision.utils as vutils


class SSuperVAEContextualAttentionalTrainer(BaseTrainer):
    def __init__(self,
                 config_disc,
                 model: SSuperVAEContextualAttentional,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer,
                 optimizer_disc,
                 scheduler=None,
                 scheduler_disc=None,
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/ssupervae_contextual_attention/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         {"optimizer": optimizer, "optimizer_disc": optimizer_disc},
                         {"scheduler": scheduler, "scheduler_disc": scheduler_disc},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.config_disc = config_disc
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.optimizer_disc = optimizer_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc

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
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0, "l1_fine": 0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss.get(k, 0))
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

            if type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                raise AssertionError("mask and mask coordinate should be available")

            _, _, interim_face_size, _ = y.shape
            target = x if y is None else y

            out, fine_faces = self.model(x,
                                         y,
                                         target,
                                         mask,
                                         mask_coordinates,
                                         interim_face_size,
                                         self.optimizer,
                                         self.optimizer_disc,
                                         self.criterion,
                                         self.config_disc.compute_g_loss,
                                         self.config_disc.l1_loss_alpha,
                                         self.config_disc.global_wgan_loss_alpha,
                                         self.config_disc.wgan_gp_lambda)

            l1_loss = nn.L1Loss()
            out['l1_fine'] = l1_loss(fine_faces, y)

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
        # TODO: do not hard code this
        fine_face_save_frequency = 5
        losses = OrderedDict()
        for counter, batch in enumerate(self.train_loader):
            if type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                raise AssertionError("mask and mask coordinate should be available")

            _, _, interim_face_size, _ = y.shape
            target = x if y is None else y

            out, fine_faces = self.model(x,
                                         y,
                                         target,
                                         mask,
                                         mask_coordinates,
                                         interim_face_size,
                                         self.optimizer,
                                         self.optimizer_disc,
                                         self.criterion,
                                         self.config_disc.compute_g_loss,
                                         self.config_disc.l1_loss_alpha,
                                         self.config_disc.global_wgan_loss_alpha,
                                         self.config_disc.wgan_gp_lambda)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.optimizer_disc.step()

            # Saving Fine Faces
            if counter % fine_face_save_frequency == 0:
                vutils.save_image(fine_faces,
                                  self.save_dir + '/results/' + self.model_name + f'fine_epoch{epoch}_samples.png',
                                  nrow=len(fine_faces),
                                  normalize=True)

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
        self.scheduler_disc.step()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.save_samples(50, self.save_dir + '/results/' + self.model_name + f'epoch{epoch}_samples.png')
        else:
            self.model.save_samples(50, self.save_dir + '/results/' + self.model_name + f'epoch{epoch}_samples.png')
                                    
        if not self.quiet:
            pbar.close()
        return losses
