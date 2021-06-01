from copy import deepcopy

import torch
from torch import nn
from torch.distributions.normal import Normal

# Models
from functional.losses.gradient_penalty import calculate_gradient_penalty
from networks.base.base_vae import BaseVAE
from networks.contextual_networks.fine_discriminators.global_discriminator import GlobalDis
from networks.contextual_networks.fine_discriminators.local_discriminator import LocalDis
from networks.contextual_networks.fine_generator.fine_generator import FineGenerator
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.intro_vae import Decoder

# Losses

# Helpers
from utils import pytorch_util as ptu


class SSuperVAEContextualAttentional(BaseVAE):

    def __init__(self,
                 backbone,
                 panel_img_size,
                 latent_dim=256,
                 embed_dim=256,
                 seq_size=3,
                 decoder_channels=[64, 128, 256, 512],
                 gen_img_size=64,
                 cnum_discriminator=32
                 ):
        super(SSuperVAEContextualAttentional, self).__init__()

        self.latent_dim = latent_dim
        self.panel_img_size = panel_img_size

        self.encoder = PlainSequentialEncoder(
            backbone, latent_dim=latent_dim, embed_dim=embed_dim, seq_size=seq_size)

        self.decoder = Decoder(
            hdim=latent_dim, channels=decoder_channels, image_size=gen_img_size)

        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )

        input_dim = 3
        # first 4 is the assumption that an image is 4 head sized
        # second 4 needs to stay same because it is needed for cnum
        # calculation
        # so below assumes that original last panel image is 256 * 256
        # TODO: con't and implement contextual forward
        cnum = panel_img_size // 4
        self.fine_generator = FineGenerator(input_dim, cnum, True, None)

        self.local_disc = LocalDis(cnum_discriminator)
        self.global_disc = GlobalDis(cnum_discriminator)

    def dis_forward(self, is_local, ground_truth, generated):
        assert ground_truth.size() == generated.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, generated], dim=0)
        batch_output = self.local_disc(batch_data) if is_local else self.global_disc(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        return real_pred, fake_pred

    def coarse_forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        x_recon = self.decode(z)
        return z, None, mu, x_recon, lg_std

    # temporary function to forward pass for fine generation
    def fine_generation_forward(self,
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
        last_panel_gts = ptu.zeros_like(x_stage_0)
        x_stage_1 = ptu.zeros_like(x_stage_0)
        for i in range(len(x)):
            last_panel = x[i, 2, :, :, :]
            output_merged_last_panel = deepcopy(last_panel)

            last_panel_face = y[i, :, :, :]
            last_panel_output_face = mu_x[i, :, :, :]

            mask_coordinates_n = mask_coordinates[i]

            original_w = abs(mask_coordinates_n[0] - mask_coordinates_n[1])
            original_h = abs(mask_coordinates_n[2] - mask_coordinates_n[3])

            # inserting original face to last panel
            modified = last_panel_face.view(1, *last_panel_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(original_w, original_h))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            # TODO: we have dimensional problem here
            #   The expanded size of the tensor (73) must match the existing size (74)
            #   at non-singleton dimension 1.
            #   Target sizes: [3, 73, 74].  Tensor sizes: [3, 74, 74]
            last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
            mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face

            # expand(torch.cuda.FloatTensor{[8, 3, 128, 128]}, size=[3, 128, 128]):
            # the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)
            x_stage_0[i, :, :, :] = last_panel * (1. - mask[i])
            last_panel_gts[i, :, :, :] = last_panel

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
        x_stage_2, offset_flow = self.fine_generator(x_stage_0, x_stage_1, mask)

        fine_faces = ptu.zeros(B, C, interim_face_size, interim_face_size)
        for i in range(len(x)):
            x_stage_2_n = x_stage_2[i, :, :, :]
            mask_coordinates_n = mask_coordinates[i]
            fine_face = x_stage_2_n[:, mask_coordinates_n[0]: mask_coordinates_n[1],
                        mask_coordinates_n[2]: mask_coordinates_n[3]]
            interpolated_fine_face = torch.nn.functional.interpolate(fine_face.view(1, *fine_face.size()),
                                                                     size=(interim_face_size, interim_face_size))
            fine_faces[i, :, :, :] = interpolated_fine_face

            # Creating x_stage_2 inpainted
            x_stage_2[i, :, :, :] = last_panel_gts[i, :, :, :] * (1. - mask[i]) + x_stage_2[i, :, :, :] * mask[i]

        return x_stage_0, x_stage_1, x_stage_2, offset_flow, fine_faces, last_panel_gts

    def forward(self,
                x,
                y,
                target,
                mask,
                mask_coordinates,
                interim_face_size,
                optimizer,
                optimizer_disc,
                criterion,
                compute_g_loss,
                l1_loss_alpha,
                global_wgan_loss_alpha,
                wgan_gp_lambda):
        if self.training:
            optimizer.zero_grad()
        z, _, mu_z, mu_x, logstd_z = self.coarse_forward(x)
        out = criterion(z.cuda(), target.cuda(), mu_z.cuda(), mu_x.cuda(), logstd_z.cuda())
        x_stage_0, \
        x_stage_1, \
        x_stage_2, \
        offset_flow, \
        fine_faces, \
        last_panel_gts = self.fine_generation_forward(x,
                                                      y,
                                                      mask,
                                                      mu_x,
                                                      mask_coordinates,
                                                      interim_face_size=interim_face_size)
        # wgan g loss
        if compute_g_loss:
            # this does not exactly match with impl because they use l1 many times in different parts of the net
            l1_loss = nn.L1Loss()
            out['l1_fine'] = l1_loss(fine_faces, y) * l1_loss_alpha

            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(is_local=True,
                                                                            ground_truth=y,
                                                                            generated=fine_faces)
            global_real_pred, global_fake_pred = self.dis_forward(is_local=False,
                                                                  ground_truth=last_panel_gts,
                                                                  generated=x_stage_2)
            # TODO: do not forget to use "backward" on this!
            out['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                            torch.mean(global_fake_pred) * global_wgan_loss_alpha

            out['loss'] = out['loss'] + out['wgan_g'] + out['l1_fine']

        if self.training:
            out['loss'].backward(retain_graph=True)
        # D part
        if self.training:
            optimizer_disc.zero_grad()
        # wgan d loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(is_local=True,
                                                                        ground_truth=y,
                                                                        generated=fine_faces)
        global_real_pred, global_fake_pred = self.dis_forward(is_local=False,
                                                              ground_truth=last_panel_gts,
                                                              generated=x_stage_2)
        # TODO: do not forget to use "backward" on this!
        out['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                        torch.mean(global_fake_pred - global_real_pred) * global_wgan_loss_alpha
        # gradients penalty loss
        local_penalty = calculate_gradient_penalty(
            self.local_disc, y, fine_faces.detach())
        global_penalty = calculate_gradient_penalty(self.global_disc,
                                                    x_stage_0, x_stage_2.detach())
        # TODO: do not forget to use "backward" on this!
        out['wgan_gp'] = local_penalty + global_penalty

        # Update D
        if self.training:
            out['d'] = out['wgan_d'] + out['wgan_gp'] * wgan_gp_lambda
            out['d'].backward()

        return out, fine_faces

    def encode(self, x):
        return self.encoder(x)

    def generate(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def sample(self, size: int, current_device: int = 0):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        return self.decode(z)

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)
