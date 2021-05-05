import torch
import torch.nn as nn
from torch import Tensor
import torchvision

# Models
from networks.sequential_encoder import SequentialEncoder
from networks.bigan import BiGAN

# Losses
from functional.losses.kl_loss import kl_loss
from functional.losses.reconstruction_loss import reconstruction_loss
from functional.losses.discrimination_loss import discrimination_loss

class SSuperGAN(nn.Module):
    
    def __init__(self, 
                 encoder_args, 
                 image_dim=64,
                 image_channel=3,
                 latent_dim=256,
                 g_hidden_size=1024,
                 d_hidden_size=1024,
                 e_hidden_size=1024, 
                 gan_type="bigan", 
                 custom_embedder=None):
        
        super(SSuperGAN, self).__init__()
        
        self.seq_encoder = SequentialEncoder(args=encoder_args, pretrained_cnn=custom_embedder)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
        if gan_type == "bigan":
            self.gan = BiGAN(
                image_dim,
                image_channel=image_channel,
                latent_dim=latent_dim,
                g_hidden_size=g_hidden_size,
                d_hidden_size=d_hidden_size,
                e_hidden_size=e_hidden_size
            )
        else:
            raise NotImplementedError     
    
    # Sample new face images from the generator of the GAN model
    def sample(self, size: int) -> Tensor:
        return self.gan.sample(size)
    
    # Encodes real face images to latent z
    def encode(self, image: Tensor) -> Tensor:
        return self.gan.encode(image)
    
    
    def forward(self, x: Tensor, y: Tensor=None) -> Tensor:
        # Sequential part
        mu, std = self.seq_encoder(x)
        z = torch.distributions.Normal(mu, std).rsample()
        # GAN part
        y_recon = self.gan.generate(z)
        z_recon = self.gan.encode(y)
        return y_recon, z_recon, (z, mu, std)

    
    def get_seq_encoder_loss(self, y, y_recon, stats):
        loss_kl = kl_loss(*stats)
        loss_recon = reconstruction_loss(y, y_recon, self.log_scale)
        return loss_kl + loss_recon
    
    def get_discriminator_loss(self, y, y_recon, z, z_recon, mu, std):
        # New random generation
        z_normal = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(std)).rsample()
        y_normal_recon = self.gan.generate(z_normal)
        # Discriminator passes
        disc_normal_fake = self.gan.discriminate(y_normal_recon, z_normal)
        disc_real = self.gan.discriminate(y, z)
        disc_fake = self.gan.discriminate(y_recon, z_recon)
        # Calculation of the discrimination loss
        return discrimination_loss(disc_real, disc_fake, disc_normal_fake)
        
    
    def get_generator_loss(self, y, y_recon, z, z_recon, mu, std):
        loss_recon = reconstruction_loss(y, y_recon, self.log_scale)
        loss_disc = self.get_discriminator_loss(y, y_recon, z, z_recon, mu, std)
        return loss_recon - loss_disc
        
        
#     # Passes 
#     def forward(self, x: Tensor, y: Tensor=None) -> Tensor:
#         # Sequential part
#         mu, std = self.seq_encoder(x)
#         z = torch.distributions.Normal(mu, std).rsample()
#         # GAN part
#         y_recon = self.gan.generate(z)
        
#         if y is None:
#             # only the generation task is applicable
#             return y_recon
#         else:
#             z_normal = torch.distributions.Normal(
#                 torch.zeros_like(mu), torch.ones_like(std)).rsample()
#             y_normal_recon = self.gan.generate(z_normal)
#             # loss calculation
#             z_recon = self.gan.encode(y)
#             disc_real = self.gan.discriminate(y, z)
#             disc_fake = self.gan.discriminate(y_recon, z_recon)
#             disc_normal_fake = self.gan.discriminate(y_normal_recon, z_normal)
            
#             loss_kl = kl_loss(z, mu, std)
#             loss_recon = reconstruction_loss(y, y_recon, self.log_scale)
#             loss_disc = discrimination_loss(disc_real, disc_fake, disc_normal_fake)
        
#             return loss_kl, loss_recon, loss_disc