from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from functional.losses.gradient_penalty import calculate_gradient_penalty
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

            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].to(ptu.device), batch[1].to(ptu.device)
            elif type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device), batch[2].to(ptu.device)
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                x, y = batch.to(ptu.device), None

            _, _, interim_face_size, _ = y.shape

            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            _, _, x_stage_2, \
            offset_flow, \
            fine_faces, last_panel_gts = self.model.fine_generation_forward(x,
                                                            y,
                                                            mask,
                                                            mu_x,
                                                            mask_coordinates,
                                                            interim_face_size=interim_face_size)
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
            batch = batch

            # This is buggy because when mask is included
            # list length is 3
            # TODO: fix this
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].to(ptu.device), batch[1].to(ptu.device)
            elif type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device), batch[2].to(ptu.device)
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                x, y = batch.to(ptu.device), None

            self.optimizer.zero_grad()
            _, _, interim_face_size, _ = y.shape
            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y

            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            # TODO: add forward pass for fine_generator
            # and add a basic loss such as l1
            # next step is to add discriminators
            # and getting loss from them
            x_stage_0, \
            x_stage_1, \
            x_stage_2, \
            offset_flow, \
            fine_faces, \
            last_panel_gts = self.model.fine_generation_forward(x,
                                                            y,
                                                            mask,
                                                            mu_x,
                                                            mask_coordinates,
                                                            interim_face_size=interim_face_size)

            # wgan g loss
            if self.config_disc.compute_g_loss:
                # this does not exactly match with impl because they use l1 many times in different parts of the net
                l1_loss = nn.L1Loss()
                out['l1_fine'] = l1_loss(fine_faces, y) * self.config_disc.l1_loss_alpha

                local_patch_real_pred, local_patch_fake_pred = self.model.dis_forward(is_local=True,
                                                                                      ground_truth=y,
                                                                                      generated=fine_faces)
                global_real_pred, global_fake_pred = self.model.dis_forward(is_local=False,
                                                                            ground_truth=last_panel_gts,
                                                                            generated=x_stage_2)
                # TODO: do not forget to use "backward" on this!
                out['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                                torch.mean(global_fake_pred) * self.config_disc.global_wgan_loss_alpha

                out['loss'] = out['loss'] + out['wgan_g'] + out['l1_fine']

            # TODO: Original Implementation Has not Retain Graph?
            #   Possible reason: params of disc is included in optimizer loss although this
            #       does not need to be the case
            out['loss'].backward(retain_graph=True)

            # D part
            # wgan d loss
            local_patch_real_pred, local_patch_fake_pred = self.model.dis_forward(is_local=True,
                                                                                  ground_truth=y,
                                                                                  generated=fine_faces)
            global_real_pred, global_fake_pred = self.model.dis_forward(is_local=False,
                                                                        ground_truth=last_panel_gts,
                                                                        generated=x_stage_2)
            # TODO: do not forget to use "backward" on this!
            out['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                            torch.mean(global_fake_pred - global_real_pred) * self.config_disc.global_wgan_loss_alpha
            # gradients penalty loss
            local_penalty = calculate_gradient_penalty(
                self.model.local_disc, y, fine_faces.detach())
            global_penalty = calculate_gradient_penalty(self.model.global_disc,
                                                        x_stage_0, x_stage_2.detach())
            # TODO: do not forget to use "backward" on this!
            out['wgan_gp'] = local_penalty + global_penalty

            # Update D
            self.optimizer_disc.zero_grad()
            out['d'] = out['wgan_d'] + out['wgan_gp'] * self.config_disc.wgan_gp_lambda
            out['d'].backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.optimizer_disc.step()

            # Saving Fine Faces
            if counter % fine_face_save_frequency == 0:
                vutils.save_image(fine_faces,
                                  self.save_dir + '/results/' + self.model_name +f'fine_epoch{epoch}_samples.png',
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
        self.model.save_samples(50, self.save_dir + '/results/' + self.model_name + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses
