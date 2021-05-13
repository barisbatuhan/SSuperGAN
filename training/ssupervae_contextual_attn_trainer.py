from collections import OrderedDict
from copy import deepcopy

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


class SSuperVAEContextualAttentionalTrainer(BaseTrainer):
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
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0, "l1_fine": 0}

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
            elif type(batch) == list and len(batch) == 4:
                x, y, mask = batch[0].to(ptu.device), batch[1].to(ptu.device), batch[2].to(ptu.device)
                mask_coordinates = ptu.get_numpy(batch[3])
            else:
                x, y = batch.to(ptu.device), None

            _, _, interim_face_size, _ = y.shape

            z, _, mu_z, mu_x, logstd_z = self.model(x)
            target = x if y is None else y
            out = self.criterion(z, target, mu_z, mu_x, logstd_z)

            x_stage_2, offset_flow, fine_faces = self.fine_generation_forward(self.model,
                                                                              x,
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
        losses = OrderedDict()
        for batch in self.train_loader:
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
            out['loss'].backward(retain_graph=True)

            # TODO: add forward pass for fine_generator
            # and add a basic loss such as l1
            # next step is to add discriminators
            # and getting loss from them
            x_stage_2, offset_flow, fine_faces = self.fine_generation_forward(self.model,
                                                                              x,
                                                                              y,
                                                                              mask,
                                                                              mu_x,
                                                                              mask_coordinates,
                                                                              interim_face_size=interim_face_size)
            l1_loss = nn.L1Loss()
            out['l1_fine'] = l1_loss(fine_faces, y)
            out['l1_fine'].backward()

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

    # temporary function to forward pass for fine generation
    def fine_generation_forward(self,
                                net,
                                x,
                                y,
                                mask,
                                mu_x,
                                mask_coordinates,
                                interim_face_size):
        # Preparing for Fine Generator
        B, S, C, W, H = x.shape
        mask = mask.view(B, 1, W, H)
        x_stage_0 = ptu.zeros(B, C, H, W)
        x_stage_1 = ptu.zeros_like(x_stage_0)
        for i in range(len(x)):
            last_panel = x[i, 2, :, :, :]
            output_merged_last_panel = deepcopy(last_panel)

            last_panel_face = y[i, :, :, :]
            last_panel_output_face = mu_x[0, :, :, :]

            mask_coordinates_n = mask_coordinates[i]

            original_w = abs(mask_coordinates_n[0] - mask_coordinates_n[1])
            original_h = abs(mask_coordinates_n[2] - mask_coordinates_n[3])

            # inserting original face to last panel
            modified = last_panel_face.view(1, *last_panel_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(original_w, original_h))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
            mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
            x_stage_0[i, :, :, :] = last_panel

            # inserting output face to last panel
            modified = last_panel_output_face.view(1, *last_panel_output_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(original_w, original_h))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            output_merged_last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
            mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
            x_stage_1[i, :, :, :] = output_merged_last_panel

        # TODO: x_stage_2 here is not in same normalized space i assume
        # we should cross check with the implementation
        # TODO: we need to normalize before visualizing
        x_stage_2, offset_flow = net.fine_generator(x_stage_0, x_stage_1, mask)

        fine_faces = ptu.zeros(B, C, interim_face_size, interim_face_size)
        for i in range(len(x)):
            x_stage_2_n = x_stage_2[i, :, :, :]
            mask_coordinates_n = mask_coordinates[i]
            fine_face = x_stage_2_n[:, mask_coordinates_n[0]: mask_coordinates_n[1],
                        mask_coordinates_n[2]: mask_coordinates_n[3]]
            interpolated_fine_face = torch.nn.functional.interpolate(fine_face.view(1, *fine_face.size()),
                                                                     size=(interim_face_size, interim_face_size))
            fine_faces[i, :, :, :] = interpolated_fine_face

        return x_stage_2, offset_flow, fine_faces
