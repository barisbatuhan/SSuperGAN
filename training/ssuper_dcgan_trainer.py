from collections import OrderedDict
from functional.losses.elbo import elbo
from functional.losses.reconstruction_loss import reconstruction_loss_distributional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from networks.models import SSuperDCGAN
from networks.models import SSuperGlobalDCGAN
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu


class SSuperDCGANTrainer(BaseTrainer):
    def __init__(self,
                 model: SSuperDCGAN,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer_encoder,
                 optimizer_generator,
                 optimized_discriminator,
                 scheduler=None,
                 quiet: bool = False,
                 grad_clip=None,
                 parallel=False,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/SSuperDCGAN/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         {
                            "optimizer_encoder": optimizer_encoder,
                            "optimizer_generator": optimizer_generator,
                            "optimizer_discriminator":optimized_discriminator},
                         
                         {"scheduler": scheduler},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.parallel = parallel

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        
        # TODO: be careful about best loss here this might override the 
        # actual best loss in case of continuation of training
        best_loss = 99999999
        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())

        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_ssuper_dcgan(epoch)
            if self.test_loader is not None:
                test_loss = self.eval_loss(self.test_loader)
            else:
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0, "disc_loss":0, "gen_loss":0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss[k])
                if k == "loss":
                    current_test_loss = test_loss[k]
                    if current_test_loss < best_loss:
                        best_loss = current_test_loss
                        if self.best_loss_action != None:
                            self.best_loss_action(self.model, best_loss)

            if self.checkpoint_every_epoch:
                self.save_checkpoint(current_loss={"train_losses": train_losses, "test_losses": test_losses},
                                     current_epoch=epoch)
            
            metric_recorder.update_metrics(train_losses, test_losses)
            metric_recorder.save_recorder()
        
        return train_losses, test_losses


    def train_ssuper_dcgan(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        
        losses = OrderedDict()
        for batch in self.train_loader:
            batch = batch
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].cuda(), batch[1].cuda()
            else:
                x, y = batch.cuda(), batch.cuda()
            
            bs = x.shape[0]
            
            # Encoder Update
            mu_z, lg_std_z = self.model(x, f="seq_encode")
            z = torch.distributions.Normal(mu_z, lg_std_z.exp()).rsample().unsqueeze(2).unsqueeze(3)
            y_recon = self.model(z, f="generate")
            out = elbo(z, y, mu_z, y_recon, lg_std_z, l1_recon=False)
            errE = out["loss"].mean()
            
            self.optimizers["optimizer_encoder"].zero_grad()
            errE.backward()

            if self.grad_clip and self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.module.seq_encoder.parameters(), self.grad_clip)
            elif self.grad_clip and not self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.seq_encoder.parameters(), self.grad_clip)
            self.optimizers["optimizer_encoder"].step()
            
            # Label set before GAN update
            labels = torch.ones(2*bs).cuda().view(-1)
            labels[bs:] = 0
            
            # Discriminator Update   
            with torch.no_grad():
                mu_z, lg_std_z = self.model(x, f="seq_encode")
                z = torch.distributions.Normal(mu_z, lg_std_z.exp()).rsample().unsqueeze(2).unsqueeze(3)
                y_recon = self.model(z.detach(), f="generate")
            
            disc_out = self.model(torch.cat([y, y_recon.detach()], dim=0), f="discriminate", local=True).view(-1)
            errD = nn.BCEWithLogitsLoss()(disc_out, labels).mean()
            out["disc_loss"] = errD
            
            self.optimizers["optimizer_discriminator"].zero_grad()
            errD.backward()
            
            if self.grad_clip and self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.module.local_discriminator.parameters(), self.grad_clip)
            elif self.grad_clip and not self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.local_discriminator.parameters(), self.grad_clip)
            self.optimizers["optimizer_discriminator"].step() 
            
            
            # Generator Update
            y_recon = self.model(z.detach(), f="generate")
            recon_loss = -reconstruction_loss_distributional(y, y_recon) / 10
            
            fake_out = self.model(y_recon, f="discriminate", local=True).view(-1)
            errG = nn.BCEWithLogitsLoss()(fake_out, labels[:bs]).mean() + recon_loss
            out["gen_loss"] = errG
            
            self.optimizers["optimizer_generator"].zero_grad()
            errG.backward()
        
            if self.grad_clip and self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.module.generator.parameters(), self.grad_clip)
            elif self.grad_clip and not self.parallel:
                torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.grad_clip)
            self.optimizers["optimizer_generator"].step()        

            desc = f'Epoch {epoch}'
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                if "gen_loss" not in losses  or "disc_loss" not in losses:
                    losses["disc_loss"] = []
                    losses["gen_loss"] = []
                             
                losses[k].append(v.item())
                
                avg = np.mean(losses[k][-50:])
                desc += f', {k} {avg:.4f}'

            if not self.quiet:
                pbar.set_description(desc)
                pbar.update(x.shape[0])

        if self.parallel:
            self.model.module.save_samples(10, self.save_dir + '/results/' + self.model_name + f'epoch{epoch}_samples.png')
        else:
            self.model.save_samples(10, self.save_dir + '/results/' + self.model_name + f'epoch{epoch}_samples.png')
        
        if not self.quiet:
            pbar.close()
        return losses
