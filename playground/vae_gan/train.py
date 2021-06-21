import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_faces import GoldenFacesDataset
from networks.models import VAEGAN
from training.vae_gan_trainer import VAEGANTrainer
from utils.config_utils import read_config, Config
from configs.base_config import *

def train(tr_data_loader, val_data_loader, config, model_name='vae_gan', cont_model=None):
    
    print("\n[INFO] Initiating training...")
    
    net = VAEGAN(
        latent_dim=config.latent_dim,
        img_size=config.img_size,
        gen_channels=config.gen_channels,
        enc_channels=config.enc_channels,
        local_disc_channels=config.local_disc_channels,
    )
    
    if config.parallel:
        net = nn.DataParallel(net).cuda() # all GPU devices available are used by default
        optimizerD_loc = optim.Adam(net.module.local_discriminator.parameters(), lr=config.local_disc_lr,
                                    betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.module.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerE = optim.Adam(net.module.encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    else:
        net = net.to(ptu.device) 
        optimizerD_loc = optim.Adam(net.local_discriminator.parameters(), lr=config.local_disc_lr,
                                    betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerE = optim.Adam(net.encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    
    
    if cont_model is not None:
        model_dict = torch.load(cont_model)
        net.load_state_dict(model_dict["model_state_dict"])
        
        optimizerD_loc.load_state_dict(model_dict["local_discriminator"])
        for g in optimizerD_loc.param_groups:
            g['lr'] = config.local_disc_lr
        
        optimizerG.load_state_dict(model_dict["generator"])
        optimizerE.load_state_dict(model_dict["encoder"])
        cont_epoch = model_dict["epoch"] + 1

    else:
        cont_epoch = None
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    # init trainer
    trainer = VAEGANTrainer(
        model=net,
        model_name=model_name,
        criterion={
            "loss_type": "basic",
            "recon_ratio": 1,
        },
        train_loader=tr_data_loader,
        test_loader=val_data_loader,
        epochs=config.train_epochs,
        optimizers={
            "generator": optimizerG,
            "local_discriminator": optimizerD_loc,
            "encoder": optimizerE,
        },
        grad_clip=config.g_clip,
        save_dir=base_dir + 'playground/vae_gan/',
        parallel=config.parallel,
        checkpoint_every_epoch=True
    )

    losses, test_losses = trainer.train_epochs(starting_epoch=cont_epoch)

if __name__ == '__main__':

    config = read_config(Config.VAEGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    cont_model = None
    
    tr_data = GoldenFacesDataset(
        golden_age_config.faces_path,
        config.img_size,
        train_mode=True,
        train_test_ratio=golden_age_config.train_test_ratio,
        augment=False,
        limit_size=-1
    )
    
    val_data = GoldenFacesDataset(
        golden_age_config.faces_path,
        config.img_size,
        train_mode=False,
        train_test_ratio=golden_age_config.train_test_ratio,
        augment=False,
        limit_size=-1
    )
    
    tr_data_loader = DataLoader(tr_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print("\nGolden Age Config:", golden_age_config)
    print("\nModel Config:", config)
    
    model_name ="vae_gan_model"
    train(tr_data_loader, val_data_loader, config, model_name, cont_model)