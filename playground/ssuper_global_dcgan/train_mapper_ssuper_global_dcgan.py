import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SSuperGlobalDCGAN
from training.ssuper_global_dcgan_mapping_trainer import SSuperGlobalDCGANMappingTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

def train(config, model_name='ssuper_global_dcgan', cont_model=None):
    
    print("[INFO] Initiating training...")
    
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
        optimizer = optim.Adam(net.module.encoder.parameters())
    else:
        net = net.to(ptu.device) 
        optimizer = optim.Adam(net.encoder.parameters())
    
    
    if cont_model is not None:
        model_dict = torch.load(cont_model)
        net.load_state_dict(model_dict["model_state_dict"])
        optimizer.load_state_dict(model_dict["optimizer"])
        cont_epoch = model_dict["epoch"] + 1

    else:
        cont_epoch = None
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    iter_cnt = 500000
    batch_size = config.batch_size
    
    # init trainer
    trainer = SSuperGlobalDCGANMappingTrainer(
        model=net,
        model_name=model_name,
        # available loss types: wgan, basic
        criterion={
            "loss_type": "basic",
        },
        epochs=config.train_epochs,
        iter_cnt=iter_cnt,
        batch_size=batch_size,
        optimizers={
            "optimizer": optimizer,
        },
        grad_clip=config.g_clip,
        save_dir=base_dir + 'playground/mapped_ssuper_global_dcgan/',
        parallel=config.parallel,
        checkpoint_every_epoch=True
    )

    losses, test_losses = trainer.train_epochs(starting_epoch=cont_epoch)
    logging.info("[INFO] Completed training!")
    return net

if __name__ == '__main__':
    
    config = read_config(Config.SSUPERGLOBALDCGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    cont_model = "playground/mapped_ssuper_global_dcgan/ckpts/lstm_ssuper_global_dcgan_model_mapped-checkpoint-epoch4.pth"
    
    if config.use_lstm:
        model_name ="lstm_ssuper_global_dcgan_model_mapped"
    else:
        model_name ="plain_ssuper_global_dcgan_model_mapped"
    
    model = train(config, model_name, cont_model)
    torch.save(model, base_dir + 'playground/ssuper_global_dcgan/results/' + "mapped_ssuper_global_dcgan_model.pth")