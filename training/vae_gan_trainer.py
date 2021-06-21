from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from networks.models import VAEGAN
from training.base_trainer import BaseTrainer
from functional.metrics.psnr import PSNR
from functional.losses.kl_loss import kl_loss 
from functional.losses.reconstruction_loss import reconstruction_loss
from functional.losses.gan_losses import StandardGAN
from configs.base_config import *
from data.augment import get_PIL_image

class VAEGANTrainer(BaseTrainer):
    def __init__(self,
                 model: VAEGAN,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizers,
                 scheduler=None,
                 quiet: bool=False,
                 grad_clip=None,
                 parallel=False,
                 save_dir=base_dir + 'playground/vae_gan/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         optimizers,
                         {"scheduler": scheduler},
                         quiet,
                         grad_clip,
                         None,
                         checkpoint_every_epoch)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.parallel = parallel
        
        if criterion["loss_type"] == "basic":
            self.local_gan_loss = StandardGAN(model, local=True)
        else:
            raise NotImplementedError

    
    def eval_model(self, epoch):
        self.model.eval()
        psnrs, l1s, iter_cnt, recon_print, bs = 0, 0, 0, False, self.test_loader.batch_size
        for x in self.test_loader:
            
            if x.shape[0] < bs:
                # disregards the last non-complete batch since it may create problems 
                # in dividing the batch to GPUs in parallel execution
                continue
            
            x = x.cuda()
           
            with torch.no_grad():
                mu_z, _ = self.model(x, f="encode")
                mu_z = mu_z.unsqueeze(-1).unsqueeze(-1)
                x_recon = self.model(mu_z, f="generate")
            psnrs += PSNR.__call__(x_recon, x, fit_range=True)
            l1s += torch.abs(x - x_recon).mean()
            iter_cnt += 1
            
            if recon_print == False:
                recon_print = True
                hsize,  wsize, h, w  = x.shape[0], 2, *x.shape[2:]
                w = (w + 100) * wsize
                h = (h + 100) * hsize
                px = 1/plt.rcParams['figure.dpi']
                f, ax = plt.subplots(hsize, wsize)
                f.set_size_inches(w*px, h*px)
                
                for bs in range(x.shape[0]):
                    ax[bs, 0].imshow(get_PIL_image(x[bs,:,:,:]))
                    ax[bs, 0].axis('off')
                    ax[bs, 1].imshow(get_PIL_image(x_recon[bs,:,:,:].clamp(-1, 1)))
                    ax[bs, 1].axis('off')
                
                ax[0, 0].title.set_text("Original")
                ax[0, 1].title.set_text("Recon")
                
                plt.savefig(self.save_dir + 'results/' + f'_epoch{epoch}_recons.png')
                
        print("\n\n-- Epoch:", epoch, 
              " --> PSNR:", psnrs.item()/iter_cnt,
              " & L1 Loss:", l1s.item()/iter_cnt,
              "\n")
        self.model.train()
        
        return {"PSNR": psnrs.item()/iter_cnt,
                "L1": l1s.item()/iter_cnt}
    
    
    def train_epochs(self, starting_epoch=None, losses={"train_losses":{}, "test_losses":{}}):
        
        train_losses, test_losses = losses["train_losses"], losses["test_losses"]

        if starting_epoch is None:
            starting_epoch = 0
        
        for epoch in range(starting_epoch, self.epochs):
            self.model.train()
            train_loss = self.train_model(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_model(epoch)
            else:
                test_loss = {"PSNR": 0, "L1":0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                train_losses[k].extend(train_loss[k])
            
            for k in test_loss.keys():
                if k not in test_losses: 
                    test_losses[k] = []
                test_losses[k].append(test_loss[k])

            if self.checkpoint_every_epoch:
                self.save_checkpoint(current_loss={"train_losses": train_losses, 
                                                   "test_losses": test_losses},
                                     current_epoch=epoch)
        
        return train_losses, test_losses


    def train_model(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        
        losses, bs = {}, self.train_loader.batch_size
        
        for x in self.train_loader:
            
            if x.shape[0] < bs:
                # disregards the last non-complete batch since it may create problems 
                # in dividing the batch to GPUs in parallel execution
                continue
            
            x, out = x.cuda(), {}
            
            # Forward Pass

            mu, log_var = self.model(x, f="encode")
            z = self.model([mu, log_var], f="reparameterize")
            x_recon = self.model(z, f="generate")
            
            sample_z = self.model(x.shape[0], f="sample_z")
            x_fake = self.model(sample_z, f="generate")
            
            # Discriminator Update
            
            errD_local = self.local_gan_loss.dis_loss(x.detach(), x_recon.detach())
            errD_fake = self.local_gan_loss.dis_loss(x.detach(), x_fake.detach())
            out["RecDisc"] = errD_local
            out["FakeDisc"] = errD_fake
            
            errD = errD_local + errD_fake

            self.optimizers["local_discriminator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errD > -4*np.log(0.8):
                errD.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="local_discriminator")
                self.optimizers["local_discriminator"].zero_grad()
            
            # Generator Update    
            
            errRecon = reconstruction_loss(x, x_recon) * self.criterion["recon_ratio"]
            errKL = kl_loss(mu, log_var)
            out["Recon"] = errRecon
            out["KL"] = errKL
            
            errG_fake = self.local_gan_loss.gen_loss(None, x_fake)
            errG_local = self.local_gan_loss.gen_loss(None, x_recon)
            
            out["RecGen"] = errG_local 
            out["FakeGen"] = errG_fake
            errG = errG_local + errG_fake + errRecon + errKL

            self.optimizers["generator"].zero_grad()
            self.optimizers["encoder"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errG > -2*np.log(0.8):
                errG.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="generator")
                    self.model(self.grad_clip, f="grad_clip", part="encoder") 
                
                self.optimizers["generator"].step() 
                self.optimizers["encoder"].step()          
            
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

        if self.parallel:
            self.model.module.save_samples(100, self.save_dir + 'results/' + f'_epoch{epoch}_samples.png')
        else:
            self.model.save_samples(100, self.save_dir + 'results/' + f'_epoch{epoch}_samples.png')
        
        if not self.quiet:
            pbar.close()
        
        return losses
