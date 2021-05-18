from collections import OrderedDict
import torch
from functional.losses.kl_loss import kl_loss
from functional.losses.reconstruction_loss import reconstruction_loss_distributional




def ssuper_dcgan_generator_loss():
    pass


def ssuper_dcgan_discriminator_loss(real,
                                    batch_size,
                                    real_label,
                                    discriminator,
                                    ):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    
    label = torch.full((batch_size,), real_label, dtype=torch.float)
    #Forward pass Discriminator
    output = discriminator(real).view(-1)
    
    error_disc_real = self.model.criterion(output,label)
    error_disc_real.backward()
    D_x = output.mean().item()

    #Train with all fake batch
    # Generate batch of latent vectors

    noise = torch.randn(batch_size, self.model.nz, 1, 1).to(ptu.device)
    # Generate fake images
    fake = self.model.generator(noise)
    label = torch.full((batch_size,), self.fake_label, dtype=torch.float).to(ptu.device)
    # Classify all fake bach with Discriminator
    output = self.model.discriminator(fake.detach()).view(-1)
    error_disc_fake = self.model.criterion(output, label)

    # Calculate Gradients for batch, accumulated (summed) with previous gradients

    error_disc_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches

    error_D = error_disc_real + error_disc_fake
    self.optimizer_disc.step()
