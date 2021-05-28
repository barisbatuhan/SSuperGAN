import enum
import math
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from functional.losses.gradient_penalty import calculate_gradient_penalty
from networks.ssuper_global_local_discriminating import SSuperGlobalLocalDiscriminating
from networks.ssupervae_contextual_attentional import SSuperVAEContextualAttentional
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
import torchvision.utils as vutils
from utils.image_utils import imshow


class GlobalLocalDiscriminatingTrainerDiscOption(enum.Enum):
    ONLY_LOCAL = 1
    ONLY_GLOBAL = 2
    GLOBAL_AND_LOCAL = 3


class GlobalLocalDiscriminatingLossType(enum.Enum):
    WGAN_GP = 1
    DC = 2


class SSuperGlobalLocalDiscriminatingTrainer(BaseTrainer):
    def __init__(self,
                 config_disc,
                 model: SSuperGlobalLocalDiscriminating,
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
                 save_dir=base_dir + 'playground/ssuper_global_local_discriminating/',
                 checkpoint_every_epoch=False,
                 in_epoch_save_frequency=50,
                 disc_option: GlobalLocalDiscriminatingTrainerDiscOption = GlobalLocalDiscriminatingTrainerDiscOption.GLOBAL_AND_LOCAL,
                 disc_loss_type: GlobalLocalDiscriminatingLossType = GlobalLocalDiscriminatingLossType.WGAN_GP
                 ):
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
        self.in_epoch_save_frequency = in_epoch_save_frequency
        self.disc_option = disc_option
        self.disc_loss_type = disc_loss_type

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        best_loss = math.inf

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_model(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_loss(self.test_loader)
            else:
                test_loss = {}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss.get(k, 0))
                if k == "loss":
                    current_test_loss = test_loss.get(k, math.inf)
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
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device), batch[2].to(ptu.device)
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                raise AssertionError("Batch should have panels, face, mask, mask_coordinates")

            target = x if y is None else y
            # it is assumed that the image is square
            _, _, interim_face_size, _ = y.shape
            # TODO: If we divert from VAE, this part should be updated
            z, _, mu_z, mu_x, logstd_z = self.model.forward(x)
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)
            pred_global, gt_global = self.model.create_global_pred_gt_images(x,
                                                                             y,
                                                                             mu_x,
                                                                             mask_coordinates)
            self.compute_generator_loss(out, y, mu_x, gt_global, pred_global)
            self.compute_discriminator_loss(out, x, y, mask, mu_x, gt_global, pred_global)

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

    # TODO: We need to design this to have dcgan_loss, wgan losses and other gan losses
    # then compare their results for our paper
    def train_model(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        losses = OrderedDict()
        for counter, batch in enumerate(self.train_loader):
            if type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device), batch[2].to(ptu.device)
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                raise AssertionError("Batch should have panels, face, mask, mask_coordinates")

            target = x if y is None else y
            # it is assumed that the image is square
            _, _, interim_face_size, _ = y.shape
            self.optimizer.zero_grad()

            # TODO: If we divert from VAE, this part should be updated
            z, _, mu_z, mu_x, logstd_z = self.model.forward(x=x)
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)
            pred_global, gt_global = self.model.create_global_pred_gt_images(x,
                                                                             y,
                                                                             mu_x,
                                                                             mask_coordinates)
            # Generator Update
            self.compute_generator_loss(out, y, mu_x, gt_global, pred_global)
            out['loss'].backward(retain_graph=True)

            # Discriminator Update
            self.optimizer_disc.zero_grad()
            self.compute_discriminator_loss(out, x, y, mask, mu_x, gt_global, pred_global)
            out['d'].backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.optimizer_disc.step()

            # Saving Fine Faces
            if counter % self.in_epoch_save_frequency == 0:
                vutils.save_image(mu_x,
                                  self.save_dir + '/results/' + self.model_name + '_' + f'in_epoch{epoch}_samples.png',
                                  nrow=len(mu_x),
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
        self.model.save_samples(50, self.save_dir + '/results/' + self.model_name + '_' + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses

    def compute_generator_loss(self,
                               out,
                               y,
                               mu_x,
                               gt_global,
                               pred_global,
                               ):
        B = y.shape[0]
        local_patch_real_pred, local_patch_fake_pred = self.model.dis_forward(is_local=True,
                                                                              ground_truth=y,
                                                                              generated=mu_x)
        global_real_pred, global_fake_pred = self.model.dis_forward(is_local=False,
                                                                    ground_truth=gt_global,
                                                                    generated=pred_global)

        if self.disc_loss_type is GlobalLocalDiscriminatingLossType.WGAN_GP:
            if self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.GLOBAL_AND_LOCAL:
                out['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                                torch.mean(global_fake_pred) * self.config_disc.global_wgan_loss_alpha
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_LOCAL:
                out['wgan_g'] = - torch.mean(local_patch_fake_pred)
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_GLOBAL:
                out['wgan_g'] = - torch.mean(global_fake_pred) * self.config_disc.global_wgan_loss_alpha
            else:
                raise NotImplementedError

            out['wgan_g'] = out['wgan_g'] * self.config_disc.gan_loss_alpha
            out['loss'] = out['loss'] + out['wgan_g']
        elif self.disc_loss_type is GlobalLocalDiscriminatingLossType.DC:
            local_patch_real_pred, local_patch_fake_pred, global_real_pred, global_fake_pred = local_patch_real_pred.view(
                B, ), local_patch_fake_pred.view(B, ), global_real_pred.view(B, ), global_fake_pred.view(B, )

            real_label = 1
            fake_label = 0
            local_label = torch.full((B,), real_label, dtype=torch.float).to(ptu.device)
            global_label = torch.full((B,), real_label, dtype=torch.float).to(ptu.device)
            dc_local_criterion = nn.BCELoss()
            dc_global_criterion = nn.BCELoss()
            if self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.GLOBAL_AND_LOCAL:
                out['dc_g_local'] = dc_local_criterion(local_patch_fake_pred, local_label)
                out['dc_g_global'] = dc_global_criterion(global_fake_pred, global_label)
                out['loss'] = out['loss'] + out['dc_g_local'] + out['dc_g_global']
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_LOCAL:
                out['dc_g_local'] = dc_local_criterion(local_patch_fake_pred, local_label)
                out['loss'] = out['loss'] + out['dc_g_local']
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_GLOBAL:
                out['dc_g_global'] = dc_global_criterion(global_fake_pred, global_label)
                out['loss'] = out['loss'] + out['dc_g_global']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def compute_discriminator_loss(self,
                                   out,
                                   x,
                                   y,
                                   mask,
                                   mu_x,
                                   gt_global,
                                   pred_global):
        B, S, C, W, H = x.shape
        mask = mask.view(B, 1, W, H)
        local_patch_real_pred, local_patch_fake_pred = self.model.dis_forward(is_local=True,
                                                                              ground_truth=y,
                                                                              generated=mu_x)
        global_real_pred, global_fake_pred = self.model.dis_forward(is_local=False,
                                                                    ground_truth=gt_global,
                                                                    generated=pred_global)
        if self.disc_loss_type is GlobalLocalDiscriminatingLossType.WGAN_GP:
            if self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.GLOBAL_AND_LOCAL:
                out['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                                torch.mean(
                                    global_fake_pred - global_real_pred) * self.config_disc.global_wgan_loss_alpha
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_LOCAL:
                out['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred)
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_GLOBAL:
                out['wgan_d'] = torch.mean(
                    global_fake_pred - global_real_pred) * self.config_disc.global_wgan_loss_alpha
            else:
                raise NotImplementedError
            # gradients penalty loss
            local_penalty = calculate_gradient_penalty(
                self.model.local_discriminator, y, mu_x.detach())
            x_stage_0 = x[:, -1, :, :, :] * (1. - mask)
            global_penalty = calculate_gradient_penalty(self.model.global_discriminator,
                                                        x_stage_0, pred_global.detach())

            out['wgan_gp'] = local_penalty + global_penalty
            out['d'] = out['wgan_d'] + out['wgan_gp'] * self.config_disc.wgan_gp_lambda
        elif self.disc_loss_type is GlobalLocalDiscriminatingLossType.DC:
            local_patch_real_pred, local_patch_fake_pred, global_real_pred, global_fake_pred = local_patch_real_pred.view(
                B, ), local_patch_fake_pred.view(B, ), global_real_pred.view(B, ), global_fake_pred.view(B, )

            real_label = 1
            fake_label = 0
            local_real_label = torch.full((B,), real_label, dtype=torch.float).to(ptu.device)
            global_real_label = torch.full((B,), real_label, dtype=torch.float).to(ptu.device)
            local_fake_label = torch.full((B,), fake_label, dtype=torch.float).to(ptu.device)
            global_fake_label = torch.full((B,), fake_label, dtype=torch.float).to(ptu.device)

            global_label = torch.cat((global_fake_label, global_real_label), 0)
            local_label = torch.cat((local_fake_label, local_real_label), 0)

            local_pred = torch.cat((local_patch_fake_pred, local_patch_real_pred), 0)
            global_pred = torch.cat((global_fake_pred, global_real_pred), 0)

            dc_local_criterion = nn.BCELoss()
            dc_global_criterion = nn.BCELoss()
            if self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.GLOBAL_AND_LOCAL:
                out['dc_d_local'] = dc_local_criterion(local_pred, local_label)
                out['dc_d_global'] = dc_global_criterion(global_pred, global_label)
                out['d'] = out['dc_d_local'] + out['dc_d_global']
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_LOCAL:
                out['dc_d_local'] = dc_local_criterion(local_pred, local_label)
                out['d'] = out['dc_d_local']
            elif self.disc_option is GlobalLocalDiscriminatingTrainerDiscOption.ONLY_GLOBAL:
                out['dc_d_global'] = dc_global_criterion(global_pred, global_label)
                out['d'] = out['dc_d_global']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
