from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from networks.models import SSuperGlobalStyleGAN2
from training.base_trainer import BaseTrainer
from functional.metrics.psnr import PSNR
from functional.losses.elbo import elbo
from functional.losses.kl_loss import *
from functional.losses.reconstruction_loss import *
from functional.losses.gan_losses import StandardGAN, WGAN_GP
from configs.base_config import *
from data.augment import get_PIL_image

class SSuperGlobalStyleGAN2Trainer(BaseTrainer):
    def __init__(self,
                 model: SSuperGlobalStyleGAN2,
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
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/ssuper_global_stylegan2/',
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
                         best_loss_action,
                         checkpoint_every_epoch)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.parallel = parallel

        if criterion["loss_type"] == "basic":
            self.local_gan_loss = StandardGAN(self.model, local=True)
            self.global_gan_loss = StandardGAN(self.model, local=False)
        else:
            raise NotImplementedError

    
    def eval_model(self, epoch):
        self.model.eval()
        psnrs, l1s, iter_cnt, recon_print, bs = 0, 0, 0, False, self.test_loader.batch_size
        for batch in self.test_loader:
            batch = batch
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = batch.cuda(), batch.cuda()
            
            if x.shape[0] < bs:
                # disregards the last non-complete batch since it may create problems 
                # in dividing the batch to GPUs in parallel execution
                continue
            
            with torch.no_grad():
                mu_z, _ = self.model(x, f="seq_encode")
                y_recon = self.model(mu_z, f="generate", map_to_w=False)
            psnrs += PSNR.__call__(y_recon, y, fit_range=True)
            l1s += torch.abs(y - y_recon).mean()
            iter_cnt += 1
            
            if recon_print == False:
                recon_print = True
                h, w = y.shape[2:]
                wsize, hsize = 2, y.shape[0]
                w = (w + 100) * wsize
                h = (h + 100) * hsize
                px = 1/plt.rcParams['figure.dpi']
                f, ax = plt.subplots(hsize, wsize)
                f.set_size_inches(w*px, h*px)
                
                for bs in range(y.shape[0]):
                    ax[bs, 0].imshow(get_PIL_image(y[bs,:,:,:]))
                    ax[bs, 0].axis('off')
                    ax[bs, 1].imshow(get_PIL_image(y_recon[bs,:,:,:].clamp(-1, 1)))
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
    
    
    def train_epochs(self, starting_epoch=None, losses={}):
        
        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

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
        
        losses = OrderedDict()
        for batch in self.train_loader:
            batch = batch
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = batch.cuda(), batch.cuda()
            
            bs, out = x.shape[0], {}
            
            # Forward Pass

            w, _ = self.model(x, f="seq_encode")
            # z = self.model([mu, log_var], f="reparameterize")
            y_recon = self.model(w, f="generate", map_to_w=False)
            
            # Local & Global Discriminator Update
            recon_global, gt_global = self.model(x, f="create_global_images", 
                                                 r_faces=y.detach(), 
                                                 f_faces=y_recon.detach(), 
                                                 mask_coordinates=coords)
            
#             recon_global, gt_global = self.model.module.create_global_images(
#                 x, 
#                 r_faces=y.detach(), 
#                 f_faces=y_recon.detach(), 
#                 mask_coordinates=coords)
            
            errD_local = self.local_gan_loss.dis_loss(y.detach(), y_recon.detach())
            errD_global = self.global_gan_loss.dis_loss(gt_global.detach(), recon_global.detach())              
            
            out["GDisc"] = errD_global
            out["LDisc"] = errD_local

            self.optimizers["local_discriminator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errD_local > -2*np.log(0.8):
                errD_local.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="local_discriminator")
                self.optimizers["local_discriminator"].zero_grad()
            
            self.optimizers["global_discriminator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errD_global > -np.log(0.8):
                errD_global.backward()           
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="global_discriminator")           
                self.optimizers["global_discriminator"].step() 
            
            # Generator Update
            
            errRecon = reconstruction_loss(y, y_recon) * self.criterion["recon_ratio"]
            out["Recon"] = errRecon
            
            # errKL = kl_loss(mu, log_var)
            # out["KL"] = errKL
            
            errG_local = self.local_gan_loss.gen_loss(None, y_recon)
            errG_global = self.global_gan_loss.gen_loss(None, recon_global)  
            
            out["Gen"] = errG_global + errG_local
            errG = self.criterion["gen_global_ratio"] * errG_global + errG_local
            errG += errRecon # + errKL

            self.optimizers["generator"].zero_grad()
            self.optimizers["seq_encoder"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errG > -2*np.log(0.8):
                errG.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="generator")
                    self.model(self.grad_clip, f="grad_clip", part="seq_encoder") 
                
                self.optimizers["generator"].step() 
                self.optimizers["seq_encoder"].step()          
            
            
            # Extra random sampling and GAN training
            
            z = self.model(bs, f="sample_z")
            y_fake = self.model(z, f="generate", map_to_w=True)
            
            errD_fake = self.local_gan_loss.dis_loss(y.detach(), y_fake.detach())
            out["Z_Disc"] = errD_fake
            self.optimizers["local_discriminator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errD_fake > -2*np.log(0.8):
                errD_fake.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="local_discriminator")
                self.optimizers["local_discriminator"].zero_grad()
            
            errG_fake = self.local_gan_loss.gen_loss(None, y_fake) 
            out["Z_Gen"] = errG_fake
            self.optimizers["generator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errG_fake > -np.log(0.8):
                errG_fake.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="generator")
                self.optimizers["generator"].step()     
            
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
