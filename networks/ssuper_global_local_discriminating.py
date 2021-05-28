from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import save_image
from networks.base.base_discriminator import BaseDiscriminator
from networks.base.base_global_local_discriminating import BaseGlobalLocalDiscriminating
from utils import pytorch_util as ptu
from copy import deepcopy


class SSuperGlobalLocalDiscriminating(BaseGlobalLocalDiscriminating):
    def __init__(self,
                 generator: nn.Module,
                 output_img_size,
                 panel_img_size,
                 local_disc_intermediate_channel_num=16,
                 global_disc_intermediate_channel_num=32,
                 create_local_disc_lambda=None,
                 create_global_disc_lambda=None):
        self.output_img_size = output_img_size
        self.panel_img_size = panel_img_size
        self.local_disc_intermediate_channel_num = local_disc_intermediate_channel_num
        self.global_disc_intermediate_channel_num = global_disc_intermediate_channel_num
        self.create_local_disc_lambda = create_local_disc_lambda
        self.create_global_disc_lambda = create_global_disc_lambda
        super().__init__(generator)

    def forward(self, **kwargs) -> List[Tensor]:
        return self.generator(**kwargs)

    def create_local_discriminator(self) -> nn.Module:
        if self.create_local_disc_lambda is not None:
            return self.create_local_disc_lambda()
        else:
            return BaseDiscriminator(spatial_dims=[self.output_img_size,
                                                   self.output_img_size],
                                     intermediate_channel_num=self.local_disc_intermediate_channel_num)

    def create_global_discriminator(self) -> nn.Module:
        if self.create_global_disc_lambda is not None:
            return self.create_global_disc_lambda()
        else:
            return BaseDiscriminator(spatial_dims=[self.panel_img_size,
                                                   self.panel_img_size],
                                     intermediate_channel_num=self.global_disc_intermediate_channel_num)

    def sample(self, size: int) -> Tensor:
        return self.generator.sample(size)

    # TODO: Check batuhans notes
    # TODO: we can return original last panel from dataset
    #  to boost performance
    def create_global_pred_gt_images(self,
                                     x,
                                     y,
                                     mu_x,
                                     mask_coordinates):
        # Preparing for Fine Generator
        B, S, C, W, H = x.shape
        last_panel_gts = ptu.zeros(B, C, H, W)
        panel_with_generation = ptu.zeros_like(last_panel_gts)
        for i in range(len(x)):
            last_panel = x[i, -1, :, :, :]
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
            last_panel[:,
            mask_coordinates_n[0]: mask_coordinates_n[1],
            mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
            last_panel_gts[i, :, :, :] = last_panel

            # inserting output face to last panel
            modified = last_panel_output_face.view(1, *last_panel_output_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(original_w, original_h))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            output_merged_last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
            mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
            panel_with_generation[i, :, :, :] = output_merged_last_panel

        return panel_with_generation, last_panel_gts

    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)
