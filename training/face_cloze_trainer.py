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

from networks.models import FaceClozeModel
from training.base_trainer import BaseTrainer
from functional.losses.kl_loss import *
from functional.losses.contrastive_loss import ContrastiveLoss
from configs.base_config import *
from data.augment import get_PIL_image

class FaceClozeTrainer(BaseTrainer):
    def __init__(self,
                 model: FaceClozeModel,
                 model_name: str,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizers,
                 scheduler=None,
                 quiet: bool=False,
                 grad_clip=None,
                 parallel=False,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/face_cloze/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         None,
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
        # self.test_loader = None
        self.parallel = parallel
        self.loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    def eval_model(self, epoch):
        self.model.eval()
        corrects, all_data = 0, 0
        with torch.no_grad():
            
            for panels, faces, labels in self.test_loader:
                all_data += panels.shape[0]
                panels, faces, labels = panels.cuda(), faces.cuda(), labels.cuda()
            
                # view change
                B, S, C, H, W = faces.shape
                faces = faces.view(-1, C, H, W)
                
                seq_emb, _ = self.model(panels, f="seq_encode")
                face_emb, _ = self.model(faces, f="encode")
                
                # view change
                face_emb = face_emb.view(B, S, -1) 
                seq_emb = seq_emb.unsqueeze(1)
                
                diff = torch.sum((seq_emb - face_emb)**2, dim=-1)
                min_diffs = torch.argmin(diff, dim=1)
                corrects += torch.sum(labels == min_diffs)
        
        print("==> Epoch:", epoch, " | Evaluated acc:", corrects.item() / all_data)     
        self.model.train()
        
        return {"Acc": corrects.item() / all_data}
        
    
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
                test_loss = {"Acc": 0}

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
        for panels, faces, labels in self.train_loader:
            panels, faces, labels = panels.cuda(), faces.cuda(), labels.cuda()
            
            # view change
            B, S, C, H, W = faces.shape
            faces = faces.view(-1, C, H, W)
            
            seq_emb, _ = self.model(panels, f="seq_encode")
            face_emb, _ = self.model(faces, f="encode")
            
            # view change
            face_emb = face_emb.view(B, S, -1)
            
            err = self.loss_fn(seq_emb, face_emb[:,-1,:], face_emb[:,0,:])
            for i in range(1, S-1):
                err += self.loss_fn(seq_emb, face_emb[:,-1,:], face_emb[:,i,:])
            err /= (S-1)
        
            self.optimizers["seq_encoder"].zero_grad()
            self.optimizers["encoder"].zero_grad()
            err.backward()
            if self.grad_clip:
                self.model(self.grad_clip, f="grad_clip", part="seq_encoder")
                self.model(self.grad_clip, f="grad_clip", part="encoder")
            self.optimizers["seq_encoder"].step()
            self.optimizers["encoder"].step()
            
            desc = f'Epoch {epoch}' + f', {"Loss"} {err.item():.4f}'

            if not self.quiet:
                pbar.set_description(desc)
                pbar.update(B)

        if not self.quiet:
            pbar.close()
        
        return losses
