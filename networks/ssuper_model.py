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

from functional.losses.elbo import elbo

class SSuperModel(nn.Module):
    
    def __init__(self, 
                 # required parameters
                 backbone="efficientnet-b5",       # options: ["resnet50", "efficientnet-bX"]
                 embed_dim: int=256,               # size of the embedding vectors of the panels taken from CNN
                 latent_dim: int=256,              # generated latent z size
                 panel_size: tuple=(300, 300),     # sizes of the panels
                 img_size: int=64,                 # generated face image size (shape is square)
                 use_lstm: bool=False,             # flag for using plain concat or lstm in sequential encoder
                 use_seq_enc: bool=True,           # Set to False of you only want to run pure generation module
                 enc_choice=None,                  # options: ["vae", None]. If "vae", then gen. should be also vae
                 gen_choice="dcgan",               # options: ["dcgan", "vae"]
                 local_disc_choice="dcgan",        # options: ["dcgan", "inpainting", None]
                 global_disc_choice="dcgan",       # options: ["dcgan", "inpainting", None]
                 gen_channels=64,                  # pass integer for DCGAN and [64, 128, 256, 512] for VAE,
                 
                 # seq. plain enc. parameters
                 seq_size: int=3,                  # number of sequential panels if plain encoder is used
                 
                 # seq. lstm enc. parameters
                 lstm_bidirectional: bool=False,   # if LSTM is used, a flag for setting bidirectionality
                 lstm_hidden: int=256,             # h and c size of the lstm hidden. If bidirectional, then h size is the half
                 lstm_dropout: float=0,            # set to 0 if num_lstm_layers == 0
                 fc_hidden_dims: list=[],          # set hidden dims if you want to add FC layers to the LSTM output  
                 fc_dropout: float=0,              # set if fc_hidden_dims is not empty
                 num_lstm_layers: float=1,         # number of layers that the LSTM encoder module includes
                 masked_first: bool=True,          # Set true to pass the masked panel image first in the LSTM
                 
                 # GAN parameters
                 local_disc_channels=64,          # same with the gen_channels but for local discr.
                 global_disc_channels=64,         # same with the gen_channels but for global discr.
                ):
        
        super(SSuperModel, self).__init__()
        
        # Input correctness checks
        assert enc_choice in ["vae", None]
        assert gen_choice in ["dcgan", "vae"]
        assert local_disc_choice in ["dcgan", "inpainting", None]
        assert global_disc_choice in ["dcgan", "inpainting", None]
        
        if type(gen_channels) == int:
            assert gen_choice == "dcgan"
        else:
            assert gen_choice == "vae"
        
        if enc_choice == "vae":
            assert gen_choice == "vae"
        
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
                hdim=latent_dim, channels=decoder_channels, image_size=img_size)
        else:
            raise NotImplementedError
        
        # Generator Module Declaration
        if gen_choice == "vae":
            self.generator = IntroVAEGenerator(
                hdim=latent_dim, channels=decoder_channels, image_size=img_size)
        
        elif gen_choice == "dcgan":
            self.generator = DCGANGenerator(img_size, 3, latent_dim, gen_channels)
        
        # Local Discriminator Module Declaration
        if local_disc_choice is None:
            self.local_discriminator = None
        elif local_disc_choice == "dcgan":
            self.local_discriminator = DCGANDiscriminator(
                img_size, 3, latent_dim, local_disc_channels)
        elif local_disc_choice == "inpainting":
            raise NotImplementedError
            
        # Global Discriminator Module Declaration
        if global_disc_choice is None:
            self.global_discriminator = None
        elif global_disc_choice == "dcgan":
            self.global_discriminator = DCGANDiscriminator(
                panel_size, 3, latent_dim, global_disc_channels)
        elif global_disc_choice == "inpainting":
            self.global_discriminator = InpaintingDiscriminator(
                panel_size, global_disc_channels)
        
        self.latent_dist = Normal(
            torch.FloatTensor([0.0]).cuda(),
            torch.FloatTensor([1.0]).cuda()
        )
        
    def forward(self, x, y=None, losses=None):

        bs = x.shape[0]
        
        if self.seq_encoder is not None or self.encoder is not None:
            
            if self.seq_encoder is not None:
                mu_z, lg_std_z = self.seq_encode(x)
                z = torch.distributions.Normal(mu_z, lg_std_z.exp()).rsample()
            
            elif self.encoder is not None:
                mu_z, lg_std_z = self.encode(x)
                z = torch.distributions.Normal(mu_z, lg_std_z.exp()).rsample()
        
        else:
            z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        
        if self.gen_choice == "dcgan":
                    z = z.unsqueeze(2).unsqueeze(3)
        
        if self.generator is not None:
            mu_x = self.generate(z)
        
        if self.seq_encoder is not None or self.encoder is not None:
            out = elbo(z, y, mu_z, mu_x, lg_std_z, l1_recon=False)
            errE = out["loss"]
            
            losses["loss"] = errE.item()
            losses["reconstruction_loss"] = out["reconstruction_loss"].item()
            losses["kl_loss"] = out["kl_loss"].item()
        else:
            errE = 0
        
        if self.local_discriminator is None and self.global_discriminator is None:
            return errE
        
        # Discriminator Loss
        labels = torch.ones(2*bs).cuda().view(-1)
        labels[bs:] = 0
    
        disc_out = self.discriminate(torch.cat([y, mu_x.detach()], dim=0), local=True).view(-1)
        errD = nn.BCELoss()(disc_out, labels)
        losses["disc_loss"] = errD.item()
        
        # Generator Loss
        gen_out = self.discriminate(mu_x, local=True).view(-1)
        errG = nn.BCELoss()(gen_out, labels[:bs]) 
        losses["gen_loss"] = errG.item() + losses["reconstruction_loss"]
        
        total_loss = errE + errD + errG
        
        if self.global_discriminator is not None:
            # TO DO: calculate global discriminator loss + generator loss 
            #        and add to the total loss
            raise NotImplementedError
        
        return total_loss
    
    
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
            w, h = abs(x1 - x2), abs(y1 - y2)

            # inserting original face to last panel
            modified = last_panel_face.view(1, *last_panel_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(h, w))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            last_panel[:, y1:y2, x1:x2] = interpolated_last_panel_face
            last_panel_gts[i, :, :, :] = last_panel

            # inserting output face to last panel
            modified = last_panel_output_face.view(1, *last_panel_output_face.size())
            interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                                 size=(h, w))
            interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
            output_merged_last_panel[:, y1:y2, x1:x2] = interpolated_last_panel_face
            panel_with_generation[i, :, :, :] = output_merged_last_panel

        return panel_with_generation, last_panel_gts
    
    
    # Samples <size> many images 
    @torch.no_grad()
    def sample(self, size :int):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
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