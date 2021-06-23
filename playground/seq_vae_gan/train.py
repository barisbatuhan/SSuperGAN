import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SeqVAEGAN
from training.seq_vae_gan_trainer import SeqVAEGANTrainer
from utils.config_utils import read_config, Config, read_config_from_path
from configs.base_config import *

def train(tr_data_loader, val_data_loader, config, model_name='seq_vae_gan', cont_model=None, base_model=None):
    
    print("\n[INFO] Initiating training...")
    
    net = SeqVAEGAN(
        backbone=config.backbone,
        embed_dim=config.embed_dim,
        latent_dim=config.latent_dim,
        img_size=config.img_size,
        use_lstm=config.use_lstm,
        gen_channels=config.gen_channels,
        enc_channels=config.enc_channels,
        local_disc_channels=config.local_disc_channels,
        global_disc_channels=config.global_disc_channels,
        seq_size=config.seq_size,
        lstm_conv=config.lstm_conv,
        lstm_bidirectional=config.lstm_bidirectional,
        lstm_hidden=config.lstm_hidden,
        lstm_dropout=config.lstm_dropout,
        fc_hidden_dims=config.fc_hidden_dims,
        fc_dropout=config.fc_dropout,
        num_lstm_layers=config.num_lstm_layers,
        masked_first=config.masked_first
    )
    
    if config.parallel:
        net = nn.DataParallel(net).cuda() # all GPU devices available are used by default
        # optimizerD_glo = optim.SGD(net.module.global_discriminator.parameters(), lr=config.global_disc_lr)
        optimizerE = optim.Adam(net.module.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    else:
        net = net.cuda()
        # optimizerD_glo = optim.SGD(net.global_discriminator.parameters(), lr=config.global_disc_lr)
        optimizerE = optim.Adam(net.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    
    if base_model is not None:
        base_model_dict = torch.load(base_model)
        net.load_state_dict(base_model_dict["model_state_dict"], strict=False)
    
    if cont_model is not None:
        model_dict = torch.load(cont_model)
        net.load_state_dict(model_dict["model_state_dict"])
        
#         optimizerD_glo.load_state_dict(model_dict["global_discriminator"])
#         for g in optimizerD_glo.param_groups:
#             g['lr'] = config.local_disc_lr
        
        optimizerE.load_state_dict(model_dict["seq_encoder"])
        cont_epoch = model_dict["epoch"] + 1

    else:
        cont_epoch = None
    
    print(net)
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    # init trainer
    trainer = SeqVAEGANTrainer(
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
            # "global_discriminator": optimizerD_glo,
            "seq_encoder": optimizerE,
        },
        grad_clip=config.g_clip,
        save_dir=base_dir + 'playground/seq_vae_gan/',
        parallel=config.parallel,
        checkpoint_every_epoch=True
    )

    losses, test_losses = trainer.train_epochs(starting_epoch=cont_epoch)

if __name__ == '__main__':

    config = read_config_from_path("configs/general_config.yaml")
    golden_age_config = read_config_from_path("configs/golden_age_config.yaml")
    cont_model = None
    base_model = "playground/vae_gan/weights/vae_gan_model-checkpoint-epoch133.pth"
    
    tr_data = GoldenPanelsDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path, 
        config.panel_size,
        config.img_size, 
        augment=False, 
        mask_val=golden_age_config.mask_val,
        mask_all=golden_age_config.mask_all,
        return_mask=golden_age_config.return_mask,
        return_mask_coordinates=golden_age_config.return_mask_coordinates,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=True,
        limit_size=-1)
    
    val_data = GoldenPanelsDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path, 
        config.panel_size,
        config.img_size, 
        augment=False, 
        mask_val=golden_age_config.mask_val,
        mask_all=golden_age_config.mask_all,
        return_mask=golden_age_config.return_mask,
        return_mask_coordinates=golden_age_config.return_mask_coordinates,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=False,
        limit_size=-1)
    
    tr_data_loader = DataLoader(tr_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    print("\nGolden Age Config:", golden_age_config)
    print("\nModel Config:", config)
    
    model_name ="seq_vae_gan_model"
    train(tr_data_loader, val_data_loader, config, model_name, cont_model, base_model)