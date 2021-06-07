""" Module implementing various loss functions """

import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis, local):
        self.dis = dis
        self.local = local

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis, local=True):

        super().__init__(dis, local)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()
        self.local = local

    def dis_loss(self, real_samps, fake_samps):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, f="discriminate", local=self.local)
        f_preds = self.dis(fake_samps, f="discriminate", local=self.local)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        preds = self.dis(fake_samps, f="discriminate", local=self.local)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False, local=True):
        super().__init__(dis, local)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged, f="discriminate", local=self.local)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, f="discriminate", local=self.local)
        real_out = self.dis(real_samps, f="discriminate", local=self.local)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps, f="discriminate", local=self.local))

        return loss


class LSGAN(GANLoss):

    def __init__(self, dis, local=True):
        super().__init__(dis, local)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((torch.mean(self.dis(real_samps, f="discriminate", local=self.local)) - 1) ** 2)
                      + (torch.mean(self.dis(fake_samps, f="discriminate", local=self.local))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((torch.mean(self.dis(fake_samps, f="discriminate", local=self.local)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, dis, local=True):
        super().__init__(dis, local)

    def dis_loss(self, real_samps, fake_samps):
        from torch.nn.functional import sigmoid
        real_scores = torch.mean(sigmoid(self.dis(real_samps, f="discriminate", local=self.local)))
        fake_scores = torch.mean(sigmoid(self.dis(fake_samps, f="discriminate", local=self.local)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps):
        from torch.nn.functional import sigmoid
        scores = torch.mean(sigmoid(self.dis(fake_samps, f="discriminate", local=self.local)))
        return 0.5 * ((scores - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, dis, local=True):
        super().__init__(dis, local)

    def dis_loss(self, real_samps, fake_samps):
        r_preds, r_mus, r_sigmas = self.dis(real_samps, f="discriminate", local=self.local)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps, f="discriminate", local=self.local)

        loss = (torch.mean(torch.nn.ReLU()(1 - r_preds)) +
                torch.mean(torch.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps):
        return -torch.mean(self.dis(fake_samps, f="discriminate", local=self.local))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis, local=True):
        super().__init__(dis, local)

    def dis_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps, f="discriminate", local=self.local)
        f_preds = self.dis(fake_samps, f="discriminate", local=self.local)
        batch = r_preds.size(0)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

        # R1
        grad_real = grad(outputs=r_preds.sum(), inputs=real_samps[-1], create_graph=True)[0]
        R1 = 0.5 * grad_real.view(batch, -1).pow(2).sum(1).mean(0)
        loss = loss + R1

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps, f="discriminate", local=self.local)
        f_preds = self.dis(fake_samps, f="discriminate", local=self.local)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(torch.nn.ReLU()(1 + r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 - f_r_diff)))


__all__ = ['LogisticGANLoss']

apply_loss_scaling = lambda x: x * torch.exp(x * np.log(2.0))
undo_loss_scaling = lambda x: x * torch.exp(-x * np.log(2.0))


class LogisticGANLoss(GANLoss):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self, dis, d_gamma=10, g_gamma=0, lod=0, local=True):
        super().__init__(dis, local)
        
        self.r1_gamma = d_gamma
        self.r2_gamma = g_gamma
        self.lod = lod

    def preprocess_image(self, images, **_unused_kwargs):
        """Pre-process images."""
        if self.lod != int(self.lod):
            downsampled_images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = F.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = self.lod - int(self.lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        
        if int(self.lod) == 0:
            return images
        
        return F.interpolate(
            images, scale_factor=(2 ** int(self.lod)), mode='nearest')

    def compute_grad_penalty(self, images, scores):
        """Computes gradient penalty."""
        image_grad = grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        """Computes loss for discriminator."""
        
        reals = self.preprocess_image(real_samps, lod=self.lod)
        reals.requires_grad = True

        # TODO: Use random labels.
        real_scores = self.dis(real_samps, f="discriminate", local=self.local)
        fake_scores = self.dis(fake_samps, f="discriminate", local=self.local)

        d_loss = F.softplus(fake_scores).mean()
        d_loss += F.softplus(-real_scores).mean()

        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        
        if self.r1_gamma:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
        
        if self.r2_gamma:
            fake_grad_penalty = self.compute_grad_penalty(fakes, fake_scores)

        return (d_loss +
                real_grad_penalty * (self.r1_gamma * 0.5) +
                fake_grad_penalty * (self.r2_gamma * 0.5))

    def gen_loss(self, real_samps, fake_samps):
        """Computes loss for generator."""
        # TODO: Use random labels.
        fake_scores =  self.dis(fake_samps, f="discriminate", local=self.local)
        g_loss = F.softplus(-fake_scores).mean()
        return g_loss


