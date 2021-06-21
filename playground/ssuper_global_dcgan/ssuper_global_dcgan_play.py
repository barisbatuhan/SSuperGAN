import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SSuperGlobalDCGAN
from training.ssuper_global_dcgan_trainer import SSuperGlobalDCGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

def train(tr_data_loader, val_data_loader, config, model_name='ssuper_global_dcgan', cont_model=None):
    
    print("\n[INFO] Initiating training...")
    
    net = SSuperGlobalDCGAN(
        backbone=config.backbone,
        embed_dim=config.embed_dim,
        latent_dim=config.latent_dim,
        img_size=config.img_size,
        use_lstm=config.use_lstm,
        gen_channels=config.gen_channels,
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
        optimizerD_glo = optim.SGD(net.module.global_discriminator.parameters(), lr=config.global_disc_lr)
        optimizerD_loc = optim.Adam(net.module.local_discriminator.parameters(), lr=config.local_disc_lr,
                                    betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.module.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerESeq = optim.Adam(net.module.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    else:
        net = net.to(ptu.device) 
        optimizerD_glo = optim.SGD(net.global_discriminator.parameters(), lr=config.global_disc_lr)
        optimizerD_loc = optim.Adam(net.local_discriminator.parameters(), lr=config.local_disc_lr,
                                    betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerESeq = optim.Adam(net.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    
    
    if cont_model is not None:
        model_dict = torch.load(cont_model)
        net.load_state_dict(model_dict["model_state_dict"])
        
        optimizerD_loc.load_state_dict(model_dict["local_discriminator"])
        for g in optimizerD_loc.param_groups:
            g['lr'] = config.local_disc_lr

        optimizerD_glo.load_state_dict(model_dict["global_discriminator"])
        for l in optimizerD_glo.param_groups:
            l['lr'] = config.global_disc_lr
        
        optimizerG.load_state_dict(model_dict["generator"])
        optimizerESeq.load_state_dict(model_dict["seq_encoder"])
        optimizerE.load_state_dict(model_dict["encoder"])
        cont_epoch = model_dict["epoch"] + 1

    else:
        cont_epoch = None
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    # init trainer
    trainer = SSuperGlobalDCGANTrainer(
        model=net,
        model_name=model_name,
        criterion={
            "loss_type": "basic",
            "gen_global_ratio": 0.005,
            "recon_ratio": 0.1,
        },
        train_loader=tr_data_loader,
        test_loader=val_data_loader,
        epochs=config.train_epochs,
        optimizers={
            "generator": optimizerG,
            "local_discriminator": optimizerD_loc,
            "global_discriminator": optimizerD_glo,
            "seq_encoder": optimizerESeq,
        },
        grad_clip=config.g_clip,
        save_dir=base_dir + 'playground/ssuper_global_dcgan/',
        parallel=config.parallel,
        checkpoint_every_epoch=True
    )

    losses, test_losses = trainer.train_epochs(starting_epoch=cont_epoch)
    logging.info("[INFO] Completed training!")
    return net

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    
    config = read_config(Config.SSUPERGLOBALDCGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    # cont_model = "playground/ssuper_global_dcgan/ckpts/lstm_ssuper_global_dcgan_model-checkpoint-epoch99.pth"
    cont_model = None
    
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
    
    
    if config.use_lstm:
        model_name ="lstm_ssuper_global_dcgan_model"
    else:
        model_name ="plain_ssuper_global_dcgan_model"
    
    model = train(tr_data_loader, val_data_loader, config, model_name, cont_model)
    torch.save(model, base_dir + 'playground/ssuper_global_dcgan/results/' + "ssuper_global_dcgan_model.pth")