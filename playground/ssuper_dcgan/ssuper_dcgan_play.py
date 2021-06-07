import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SSuperDCGAN
from training.ssuper_dcgan_trainer import SSuperDCGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

def save_best_loss_model(model_name, model, best_loss):
    print("[INFO] Current Best Loss:", best_loss)
    torch.save(model, base_dir + 'playground/ssuper_dcgan/weights/' + model_name + ".pth")


def train(data_loader, config, dataset, model_name='ssuper_dcgan',):
    
    print("[INFO] Initiating training...")
    
    net = SSuperDCGAN(backbone=config.backbone,
                      embed_dim=config.embed_dim,
                      latent_dim=config.latent_dim,
                      img_size=config.img_size,
                      use_lstm=config.use_lstm,
                      gen_channels=config.gen_channels,
                      local_disc_channels=config.local_disc_channels,
                      seq_size=config.seq_size,
                      lstm_conv=config.lstm_conv,
                      lstm_bidirectional=config.lstm_bidirectional,
                      lstm_hidden=config.lstm_hidden,
                      lstm_dropout=config.lstm_dropout,
                      fc_hidden_dims=config.fc_hidden_dims,
                      fc_dropout=config.fc_dropout,
                      num_lstm_layers=config.num_lstm_layers,
                      masked_first=config.masked_first)
    
    if config.parallel:
        net = nn.DataParallel(net).cuda() # all GPU devices available are used by default
        optimizerE = optim.Adam(net.module.seq_encoder.parameters(), lr=config.lr)
        optimizerD = optim.Adam(net.module.local_discriminator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.module.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    else:
        net = net.to(ptu.device) 
        optimizerE = optim.Adam(net.seq_encoder.parameters(), lr=config.lr)
        optimizerD = optim.Adam(net.local_discriminator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerG = optim.Adam(net.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    # init trainer
    trainer = SSuperDCGANTrainer(model=net,
                                 model_name=model_name,
                                 criterion=nn.BCELoss,
                                 train_loader=data_loader,
                                 test_loader=None,
                                 epochs=config.train_epochs,
                                 optimizer_encoder=optimizerE,
                                 optimizer_generator= optimizerG,
                                 optimized_discriminator= optimizerD,
                                 grad_clip=config.g_clip,
                                 best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l),
                                 save_dir=base_dir + 'playground/ssuper_dcgan/',
                                 parallel=config.parallel,
                                 checkpoint_every_epoch=True)

    losses, test_losses = trainer.train_epochs()
    logging.info("[INFO] Completed training!")
    return net

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    
    config = read_config(Config.SSUPERDCGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    cont_epoch = -1
    cont_model = None # "playground/ssupervae/weights/model-18.pth"
    
    data = GoldenPanelsDataset(golden_age_config.panel_path,
                               golden_age_config.sequence_path, 
                               config.panel_size,
                               config.img_size, 
                               augment=False, 
                               mask_val=golden_age_config.mask_val,
                               mask_all=golden_age_config.mask_all,
                               return_mask=False,
                               return_mask_coordinates=False,
                               train_test_ratio=golden_age_config.train_test_ratio,
                               train_mode=True,
                               limit_size=-1)
    
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    if config.use_lstm:
        model_name ="lstm_ssuper_dcgan_model"
    else:
        model_name ="plain_ssuper_dcgan_model"
    
    model = train(data_loader, config, model_name)
    torch.save(model, base_dir + 'playground/ssuper_dcgan/results/' + "ssuper_dcgan_model.pth")
        
        