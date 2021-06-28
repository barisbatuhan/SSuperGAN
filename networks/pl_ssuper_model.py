import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torchvision
from torchvision.utils import save_image

import copy
import numpy as np
from copy import deepcopy

# Models
from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder

from networks.encoder.introvae_encoder import IntroVAEEncoder

from networks.generator.dcgan_generator import DCGANGenerator
from networks.generator.introvae_generator import IntroVAEGenerator

from networks.discriminator.dcgan_discriminator import DCGANDiscriminator
from networks.discriminator.inpainting_discriminator import InpaintingDiscriminator

import pytorch_lightning as pl


class SSuperModel(pl.LightningModule):

    def __init__(self,
                 # params to run with pl
                 save_dir,
                 model_name,
                 # required parameters
                 backbone="efficientnet-b5",  # options: ["resnet50", "efficientnet-bX"]
                 embed_dim: int = 256,  # size of the embedding vectors of the panels taken from CNN
                 latent_dim: int = 256,  # generated latent z size
                 panel_size: tuple = (300, 300),  # sizes of the panels
                 img_size: int = 64,  # generated face image size (shape is square)
                 use_lstm: bool = False,  # flag for using plain concat or lstm in sequential encoder
                 use_seq_enc: bool = True,  # Set to False of you only want to run pure generation module
                 enc_choice=None,  # options: ["vae", None]. If "vae", then gen. should be also vae,
                 # "stylegan" provides the mapping module from z -> w
                 gen_choice="dcgan",  # options: ["dcgan", "vae"]
                 local_disc_choice="dcgan",  # options: ["dcgan", inpainting", None]
                 global_disc_choice="dcgan",  # options: ["dcgan", "inpainting", None]
                 gen_channels=64,  # pass integer for DCGAN
                 enc_channels=[64, 128, 256, 512, 512, 512],  # encoder channels with default values of IntroVAE

                 gen_norm="batch",  # options: ["batch", "instance"] and "layer" if DCGAN
                 enc_norm="batch",  # options: ["batch", "instance"]
                 disc_norm="batch",  # options: ["batch", "instance"] and "layer" if DCGAN

                 # seq. plain enc. parameters
                 seq_size: int = 3,  # number of sequential panels if plain encoder is used

                 # seq. lstm enc. parameters
                 lstm_conv: bool = False,  # if ConvLSTM module is wanted to be used instead of normal LSTM
                 lstm_bidirectional: bool = False,  # if LSTM is used, a flag for setting bidirectionality
                 lstm_hidden: int = 256,  # h and c size of the lstm hidden. If bidirectional, then h size is the half
                 lstm_dropout: float = 0,  # set to 0 if num_lstm_layers == 0
                 fc_hidden_dims: list = [],  # set hidden dims if you want to add FC layers to the LSTM output
                 fc_dropout: float = 0,  # set if fc_hidden_dims is not empty
                 num_lstm_layers: float = 1,  # number of layers that the LSTM encoder module includes
                 masked_first: bool = True,  # Set true to pass the masked panel image first in the LSTM

                 # GAN parameters
                 local_disc_channels=64,  # same with the gen_channels but for local discr.
                 global_disc_channels=64,
                 # same with the gen_channels but for global discr.
                 **kwargs
                 ):

        self.save_dir = save_dir
        self.model_name = model_name

        super(SSuperModel, self).__init__()
        self.save_hyperparameters()
        # Input correctness checks
        assert enc_choice in ["vae", None]
        assert gen_choice in ["dcgan", "vae", None]
        assert local_disc_choice in ["dcgan", "inpainting", None]
        assert global_disc_choice in ["dcgan", "inpainting", None]

        self.latent_dim = latent_dim
        self.gen_choice = gen_choice
        self.local_disc_choice = local_disc_choice
        self.global_disc_choice = global_disc_choice
        self.enc_choice = enc_choice
        # Sequential Encoder Declaration
        if not use_seq_enc:
            self.seq_encoder = None

        elif not use_lstm:
            self.seq_encoder = PlainSequentialEncoder(backbone,
                                                      latent_dim=latent_dim,
                                                      embed_dim=embed_dim,
                                                      seq_size=seq_size)
        else:
            self.seq_encoder = LSTMSequentialEncoder(backbone,
                                                     latent_dim=latent_dim,
                                                     embed_dim=embed_dim,
                                                     conv_lstm=lstm_conv,
                                                     lstm_hidden=lstm_hidden,
                                                     lstm_dropout=lstm_dropout,
                                                     lstm_bidirectional=lstm_bidirectional,
                                                     fc_hidden_dims=fc_hidden_dims,
                                                     fc_dropout=fc_dropout,
                                                     num_lstm_layers=num_lstm_layers,
                                                     masked_first=masked_first)

        # Encoder Module Declaration
        if enc_choice is None:
            self.encoder = None
        elif enc_choice == "vae":
            self.encoder = IntroVAEEncoder(
                hdim=latent_dim, channels=enc_channels, image_size=img_size, normalize=enc_norm)
        else:
            raise NotImplementedError

        # Generator Module Declaration
        if gen_choice == "vae":
            self.generator = IntroVAEGenerator(
                hdim=latent_dim, channels=enc_channels, image_size=img_size, normalize=gen_norm)

        elif gen_choice == "dcgan":
            self.generator = DCGANGenerator(img_size, 3, latent_dim, gen_channels, normalize=gen_norm)

        # Local Discriminator Module Declaration
        if local_disc_choice is None:
            self.local_discriminator = None
        elif local_disc_choice == "dcgan":
            self.local_discriminator = DCGANDiscriminator(
                img_size, 3, latent_dim, local_disc_channels, normalize=disc_norm)
        elif local_disc_choice == "inpainting":
            self.local_discriminator = InpaintingDiscriminator(
                (img_size, img_size), 3, local_disc_channels, normalize=disc_norm)

        # Global Discriminator Module Declaration
        if global_disc_choice is None:
            self.global_discriminator = None
        elif global_disc_choice == "dcgan":
            self.global_discriminator = DCGANDiscriminator(
                panel_size, 3, latent_dim, global_disc_channels, normalize=disc_norm)
        elif global_disc_choice == "inpainting":
            self.global_discriminator = InpaintingDiscriminator(
                panel_size, 3, global_disc_channels, normalize=disc_norm)

    def forward(self, x, f=None, **kwargs):
        func = getattr(self, f)
        return func(x, **kwargs)

    # Returns mu, lg_std
    def seq_encode(self, x):
        return self.seq_encoder(x)

    # Returns mu, lg_std
    def encode(self, x):
        return self.encoder(x)

    # Returns the generated image in the range [-1, 1]
    def generate(self, z, clamp=False):
        out = self.generator(z)
        if clamp:
            out = torch.clamp(out, min=-1, max=1)
        return out

    # Returns a float in the range of [0, 1] depending on the success of the discriminator
    def discriminate(self, x, local=True):
        if local:
            return self.local_discriminator(x)
        else:
            return self.global_discriminator(x)

    def reparameterize(self, data):
        # creates z from mean and log variance
        mu, logvar = data
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = torch.Tensor(eps).cuda()
        return eps.mul(std).add_(mu)

    def grad_clip(self, gclip, part=None):

        if part == "local_discriminator":
            torch.nn.utils.clip_grad_norm_(self.local_discriminator.parameters(), gclip)
        elif part == "global_discriminator":
            torch.nn.utils.clip_grad_norm_(self.global_discriminator.parameters(), gclip)
        elif part == "generator":
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), gclip)
        elif part == "seq_encoder":
            torch.nn.utils.clip_grad_norm_(self.seq_encoder.parameters(), gclip)
        elif part == "encoder":
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), gclip)
        else:
            raise NotImplementedError

    def create_global_images(self, panels, r_faces, f_faces, mask_coordinates):
        # Preparing for Fine Generator
        B, S, C, W, H = panels.shape
        last_panel_gts = torch.zeros(B, C, H, W).cuda()
        panel_with_generation = torch.zeros_like(last_panel_gts).cuda()
        for i in range(len(panels)):
            last_panel = panels[i, -1, :, :, :]
            output_merged_last_panel = deepcopy(last_panel)
            last_panel_face = r_faces[i, :, :, :]
            last_panel_output_face = f_faces[i, :, :, :]

            y1, y2, x1, x2 = mask_coordinates[i]
            x1, y1 = max(0, x1), max(0, y1)
            w, h = abs(x1 - x2), abs(y1 - y2)

            # inserting original face to last panel
            modified = last_panel_face.view(1, *last_panel_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(h, w))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0, :, :, :]
            _, p_h, p_w = interpolated_last_panel_face.shape
            _, l_h, l_w = last_panel.shape

            sel_h = min(y1 + p_h, l_h)
            sel_w = min(x1 + p_w, l_w)

            last_panel[:, y1:sel_h, x1:sel_w] = interpolated_last_panel_face[:, :sel_h - y1, :sel_w - x1]
            last_panel_gts[i, :, :, :] = last_panel

            # inserting output face to last panel
            modified = last_panel_output_face.view(1, *last_panel_output_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(h, w))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0, :, :, :]

            _, p_h, p_w = interpolated_last_panel_face.shape
            _, l_h, l_w = output_merged_last_panel.shape

            sel_h = min(y1 + p_h, l_h)
            sel_w = min(x1 + p_w, l_w)

            output_merged_last_panel[:, y1:sel_h, x1:sel_w] = interpolated_last_panel_face[:, :sel_h - y1, :sel_w - x1]
            panel_with_generation[i, :, :, :] = output_merged_last_panel

        return panel_with_generation, last_panel_gts

    @torch.no_grad()
    def sample_z(self, size: int):
        return torch.zeros(size, self.latent_dim).normal_(0, 1).cuda()

        # Samples <size> many images

    @torch.no_grad()
    def sample(self, size: int):
        z = self(size, f="sample_z")
        return self.generate(z, clamp=True)

    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)

    # Reconstructs the image given the panel images or initial image
    @torch.no_grad()
    def reconstruct(self, x, seq_encoder=True):
        mu, _ = self.seq_encode(x) if seq_encoder else self.encode(x)
        return self.generate(mu, clamp=True)

    def configure_optimizers(self):
        raise NotImplementedError

    def optimizer_step(self, *args, **kwargs):
        raise NotImplementedError

    def _calculate_loss(self, batch, mode):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
