from collections import OrderedDict
from functional.losses.elbo import elbo
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from networks.base.base_vae import BaseVAE
from networks.generic_vae import GenericVAE
from networks.ssuper_dcgan import SSuperDCGAN
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
        """
        self.optimizer = {
                        "optimizer_encoder": optimizer_encoder,
                        "optimizer_generator": optimizer_generator,
                        "optimized_discriminator":optimized_discriminator
                        },
        self.scheduler = scheduler"""


        

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        # TODO: becareful about best loss here this might override the actual best loss
        #  in case of continuation of training
        best_loss = 99999999

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())
        #torch.autograd.set_detect_anomaly(True)
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
                self.save_checkpoint(current_loss=
                {
                    "train_losses": train_losses,
                    "test_losses": test_losses
                },
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
                # Shape is 
                #torch.Size([BATCH_SIZE, 3, 3, 300, 300])
                #torch.Size([BATCH_SIZE, 3, 64, 64])
            else:
                x, y = batch.cuda(), None
            
            
            
            self.optimizers["optimizer_encoder"].zero_grad()
            self.optimizers["optimizer_discriminator"].zero_grad()
            self.optimizers["optimizer_generator"].zero_grad()
            
            z, _, mu_z, mu_x, logstd_z = self.model(x)
            
            target = x if y is None else y


            out = elbo(z, target, mu_z, mu_x, logstd_z, l1_recon=True)

            reconstruction_loss = out["reconstruction_loss"]
            kl_loss = out["kl_loss"]
            total_loss = out["loss"]


            # UPDATE ENCODER
            total_loss.backward(retain_graph=True)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.grad_clip)
            
            
            
            # Discriminator Loss with Real Dtat
            #self.model.dcgan.generator
            real_label = 1
            fake_label = 0
            bs = x.shape[0]
            label = torch.full((bs,), real_label, dtype=torch.float, device=ptu.device)
            output = self.model.dcgan.discriminator(y).view(-1)
            #print("OUTPUT DISC ",output, "SHAPE ",output.shape, "label ",label, "label shape ",label.shape)
            errD_real = nn.BCELoss()(output, label)
            errD_real.backward()
            D_x = output.mean().item()


            #Train with all-fake batch
            # mu_x --> Generated Images
            label2 = torch.full((bs,), fake_label, dtype=torch.float, device=ptu.device)
            output = self.model.dcgan.discriminator(mu_x.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = nn.BCELoss()(output, label2)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizers["optimizer_discriminator"].step()


            


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            self.model.dcgan.generator.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.model.dcgan.discriminator((mu_x)).view(-1)
            # Calculate G's loss based on this output
            errG = nn.BCELoss()(output, label) + reconstruction_loss
            # Calculate gradients for G
            errG.backward()
        
            # Update G
            #optimizerG.step()
            

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.dcgan.generator.parameters(), self.grad_clip)
                
            self.optimizers["optimizer_generator"].step()
            self.optimizers["optimizer_encoder"].step()
            
    
        
            #self.optimizers["optimizer_encoder"].step()
            desc = f'Epoch {epoch}'
            out["disc_loss"] =  errD
            out["gen_loss"] = errG
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

        #self.scheduler.step()
        self.model.save_samples(100, self.save_dir + '/results/' + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses
