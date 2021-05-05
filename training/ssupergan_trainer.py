from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.ssupergan import SSuperGAN
from utils import pytorch_util as ptu
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from configs.base_config import *

# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 256]], which is output 0 of TBackward, is at version 2; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).


class SSuperGANTrainer:
    def __init__(self,
                 model: SSuperGAN,
                 data_loader,
                 epochs: int,
                 optimizer_disc,
                 optimizer_gen,
                 optimizer_seq_enc,
                 scheduler_disc=None,
                 scheduler_gen=None,
                 scheduler_seq_enc=None,
                 gen_steps=1,
                 disc_steps=1,
                 quiet=False,
                 save_dir="playground/ssupergan/"
                ):
        
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.save_dir = save_dir
        self.optimizer_disc = optimizer_disc
        self.optimizer_gen = optimizer_gen
        self.optimizer_seq_enc = optimizer_seq_enc
        self.steps_gen = gen_steps
        self.steps_disc = disc_steps
        self.scheduler_disc = scheduler_disc
        self.scheduler_gen = scheduler_gen
        self.scheduler_seq_enc = scheduler_seq_enc
    
    def train_model(self):
        self.model.train()
        metric_recorder = MetricRecorder(save_dir=self.save_dir + '/results/')
        losses = OrderedDict()
        losses["seq_enc"], losses["gen"], losses["disc"] = [], [], []
        
        logging.info("======= TRAINING STARTS =======")
        
        for epoch in range(self.epochs):
            
            pbar = tqdm(total=len(self.data_loader.dataset))
            
            for iterno, (x, y) in enumerate(self.data_loader):
                x, y = x.to(ptu.device), y.to(ptu.device)
                batch_size = x.shape[0]              
                
                # Sequential Encoder (CNN + LSTM) update
                y_recon, z_recon, (z, mu, std) = self.model(x, y)
                loss_seq_enc = self.model.get_seq_encoder_loss(y, y_recon, (z, mu, std))
                enc_val = loss_seq_enc.item()
                self.optimizer_seq_enc.zero_grad()
                loss_seq_enc.backward(retain_graph=True)
                self.optimizer_seq_enc.step()
                
                # Discriminator update
                if iterno % self.steps_disc == 0:
                    y_recon, z_recon, (z, mu, std) = self.model(x, y)
                    loss_disc = self.model.get_discriminator_loss(y, y_recon, z, z_recon, mu, std)
                    disc_val = loss_disc.item()
                    self.optimizer_disc.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    self.optimizer_disc.step()
                else:
                    disc_val = -1
                
                # Generator & Encoder update
                if iterno % self.steps_gen == 0:
                    y_recon, z_recon, (z, mu, std) = self.model(x, y)
                    loss_gen = self.model.get_generator_loss(y, y_recon, z, z_recon, mu, std)
                    gen_val = loss_gen.item()
                    self.optimizer_gen.zero_grad()
                    loss_gen.backward()
                    self.optimizer_gen.step()
                else:
                    gen_val = -1
                
                # Loss dict update
                losses["seq_enc"].append(enc_val)
                losses["gen"].append(gen_val)
                losses["disc"].append(disc_val)
                
                # Logger and tqdm updates
                desc  = f'Epoch {epoch}'
                desc += f' --> Seq. Enc.: {enc_val:.4f}'
                desc += f' | Disc.: {disc_val:.4f}'
                desc += f' | Gen.: {gen_val:.4f}'
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
            if self.scheduler_seq_enc is not None:
                self.scheduler_seq_enc.step()

            # saving the model weights after each epoch
            if self.save_dir is not None:
                torch.save(self.model.state_dict(), self.save_dir + 'weights/')
            
            # creating and saving images after each epoch
            if self.save_dir is not None:
                self.model.save_samples(10, self.save_dir + 'samples/ + 'f'epoch{epoch}_samples.png')  
        
        return losses