from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import pytorch_util as ptu
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from configs.base_config import *

from networks.models import DCGAN

class DCGANTrainer:
    def __init__(self,
                 model:DCGAN,
                 data_loader,
                 epochs: int,
                 optimizer_disc,
                 optimizer_gen,
                 dataset_name,
                 scheduler_disc=None,
                 scheduler_gen=None,
                 gen_steps=1,
                 disc_steps=1,
                 quiet=False,
                 real_label = 1,
                 fake_label = 0,
                 save_dir="playground/dcgan",
                 
                ):
        
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.save_dir = save_dir
        self.optimizer_disc = optimizer_disc
        self.optimizer_gen = optimizer_gen
        self.steps_gen = gen_steps
        self.steps_disc = disc_steps
        self.scheduler_disc = scheduler_disc
        self.scheduler_gen = scheduler_gen
        self.real_label = real_label
        self.fake_label =  fake_label
    
    def train_epochs(self):

        """ 
        IDEA:
        construct different mini-batches for real and fake” images, 
        and also adjust G’s objective function to maximize logD(G(z)). 

        Discriminator:
        Practically:
        Maximize ---> log(D(x)) + log(1 -D(G(z)))
        Discriminator should say 1(real) for x (real sample batch) and 
        Discriminator should say 0(fake) for G(z) generated samples
        to Maximize this objective

        Generator:
        Practically we want to maximize log(D(G(z)))

        Since the Loss is BCE Loss
        For discriminator giving labels as real help us to log(x) in BCE loss rather than log(1-x) part

        BCE : Loss(x,y) = [yn*logxn + (1-yn)*log(1-xn)]
        

        """
        self.model.to(ptu.device)
        self.model.train()
        metric_recorder = MetricRecorder(save_dir=self.save_dir + 'results/')
        losses = OrderedDict()
        losses["seq_enc"], losses["gen"], losses["disc"] = [], [], []
        
        logging.info("======= TRAINING STARTS =======")
        
        for epoch in range(self.epochs):
            
            pbar = tqdm(total=len(self.data_loader.dataset))
            
            for iterno, x in enumerate(self.data_loader):
                
                x = x.to(ptu.device)
                batch_size = x.shape[0]

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.model.discriminator.zero_grad()
                real = x.to(ptu.device)
                label = torch.full((batch_size,), self.real_label, dtype=torch.float).to(ptu.device)
                #Forward pass Discriminator
                output = self.model.discriminator(real).view(-1)
                
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

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.model.generator.zero_grad()
                label = torch.full((batch_size,), self.real_label, dtype=torch.float).to(ptu.device)
                output = self.model.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                error_G = self.model.criterion(output, label)

                error_G.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer_gen.step()


                # Loss dict update
                
                losses["gen"].append(error_G.item())
                losses["disc"].append(error_D.item())
                
                # Logger and tqdm updates
                desc  = f'Epoch {epoch}'
                desc += f' | Disc.: {error_D:.4f}'
                desc += f' | Gen.: {error_G:.4f}'
                logging.info(desc)
                pbar.set_description(desc)
                pbar.update(batch_size)
                
            # close the progress bar
            pbar.close()
            
            # save training metrics
            metric_recorder.update_metrics(train=losses, test=None)
            metric_recorder.save_recorder()
            
            # step the learning rate
            if self.scheduler_disc is not None:
                self.scheduler_disc.step()
            if self.scheduler_gen is not None:
                self.scheduler_gen.step()
            

            # saving the model weights after each epoch
            if self.save_dir is not None:
                torch.save(self.model.state_dict(), self.save_dir + f'weights/dcgan_model_{epoch}.pth')
            
            # creating and saving images after each epoch
            if self.save_dir is not None:
                self.model.save_samples(100, self.save_dir + 'samples/ + 'f'epoch{epoch}_samples.png')  
        
        return losses