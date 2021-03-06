from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from networks.models import SeqVAEGAN
from training.base_trainer import BaseTrainer
from functional.metrics.psnr import PSNR
from functional.losses.reconstruction_loss import reconstruction_loss
from functional.losses.gan_losses import StandardGAN
from configs.base_config import *
from data.augment import get_PIL_image

class SeqVAEGANTrainer(BaseTrainer):
    def __init__(self,
                 model: SeqVAEGAN,
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
                 save_dir=base_dir + 'playground/seq_vae_gan/',
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
            self.global_gan_loss = StandardGAN(model, local=False)
        else:
            raise NotImplementedError

    
    def eval_model(self, epoch):
        self.model.eval()
        psnrs, l1s, iter_cnt, recon_print, bs = 0, 0, 0, False, self.test_loader.batch_size
        
        for batch in self.train_loader:
            
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = None, batch.cuda()
            
            if x.shape[0] < bs:
                # disregards the last non-complete batch since it may create problems 
                # in dividing the batch to GPUs in parallel execution
                continue
           
            with torch.no_grad():
                mu_z, _ = self.model(x, f="seq_encode")
                y_recon = self.model(mu_z, f="generate")
            psnrs += PSNR.__call__(y_recon, y, fit_range=True)
            l1s += torch.abs(y - y_recon).mean()
            iter_cnt += 1
            
            if recon_print == False:
                recon_print = True
                hsize,  wsize, h, w  = y.shape[0], 2, *y.shape[2:]
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
        
        if self.parallel:
            self.model.module.encoder.eval()
        else:
            self.model.encoder.eval()
        
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        
        losses, bs = {}, self.train_loader.batch_size
        
        for batch in self.train_loader:
            
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            elif type(batch) == list and len(batch) == 3:
                x, y, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            elif type(batch) == list and len(batch) == 4:
                x, y, mask, coords = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            else:
                x, y = None, batch.cuda()
            
            if x.shape[0] < bs:
                # disregards the last non-complete batch since it may create problems 
                # in dividing the batch to GPUs in parallel execution
                continue
            
            out = {}
            
            # Forward Pass
            
            seq_mu, seq_log_var = self.model(x, f="seq_encode")
            y_recon = self.model(seq_mu, f="generate")
            
            # Local & Global Discriminator Update
#             recon_global, gt_global = self.model(x, f="create_global_images", 
#                                                  r_faces=y.detach(), 
#                                                  f_faces=y_recon.detach(), 
#                                                  mask_coordinates=coords)
            
            errD_local = self.local_gan_loss.dis_loss(y.detach(), y_recon.detach())
            out["LDisc"] = errD_local
            
#             errD_global = self.global_gan_loss.dis_loss(gt_global.detach(), recon_global.detach())              
#             out["GDisc"] = errD_global

            self.optimizers["local_discriminator"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errD_local > -2*np.log(0.8):
                errD_local.backward()
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="local_discriminator")
                self.optimizers["local_discriminator"].zero_grad()
            
#             self.optimizers["global_discriminator"].zero_grad()
#             if self.criterion["loss_type"] != "basic" or errD_global > -2*np.log(0.8):
#                 errD_global.backward()           
#                 if self.grad_clip:
#                     self.model(self.grad_clip, f="grad_clip", part="global_discriminator")           
#                 self.optimizers["global_discriminator"].step() 
            
            # Generator & Seq. Encoder Update
            
            with torch.no_grad():
                mu, log_var = self.model(y, f="encode")
            
            errRecon_mu = reconstruction_loss(mu.detach(), seq_mu.detach())
            errRecon_logvar = reconstruction_loss(log_var.detach(), seq_log_var.detach())
            out["EncRecon"] = errRecon_mu + errRecon_logvar
            
            errRecon = reconstruction_loss(y, y_recon) * self.criterion["recon_ratio"]
            out["Recon"] = errRecon
            
            errG_local = self.local_gan_loss.gen_loss(None, y_recon)
#             errG_global = self.global_gan_loss.gen_loss(None, recon_global)  
            
            out["LGen"] = errG_local
#             out["GGen"] = errG_global
            
            errG = errG_local # + self.criterion["gen_global_ratio"] * errG_global
            errG += errRecon + errRecon_mu + errRecon_logvar

            self.optimizers["generator"].zero_grad()
            self.optimizers["seq_encoder"].zero_grad()
            if self.criterion["loss_type"] != "basic" or errG > -2*np.log(0.8):
                
                errG.backward()
                
                if self.grad_clip:
                    self.model(self.grad_clip, f="grad_clip", part="generator")
                    self.model(self.grad_clip, f="grad_clip", part="seq_encoder") 
                
                self.optimizers["generator"].step() 
                self.optimizers["seq_encoder"].step()     
            
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
