import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_face_cloze import GoldenFaceClozeDataset
from networks.models import FaceClozeModel
from training.face_cloze_trainer import FaceClozeTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

def train(tr_data_loader, val_data_loader, config, model_name='face_cloze_model', cont_model=None):
    
    print("[INFO] Initiating training...")
    
    net = FaceClozeModel(
        backbone=config.backbone,
        embed_dim=config.embed_dim,
        latent_dim=config.latent_dim,
        img_size=config.img_size,
        use_lstm=config.use_lstm,
        gen_channels=[64, 128, 256, 512],
        local_disc_channels=config.local_disc_channels,
        global_disc_channels=config.global_disc_channels,
        seq_size=config.seq_size,
        lstm_conv=False,
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
        optimizerSeq = optim.Adam(net.module.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerEnc = optim.Adam(net.module.encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    else:
        net = net.to(ptu.device) 
        optimizerSeq = optim.Adam(net.seq_encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
        optimizerEnc = optim.Adam(net.encoder.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))
    
    if cont_model is not None:
        model_dict = torch.load(cont_model)
        net.load_state_dict(model_dict["model_state_dict"])
        
        optimizerSeq.load_state_dict(model_dict["seq_encoder"])
        for g in optimizerD_loc.param_groups:
            g['lr'] = config.local_disc_lr

        optimizerEnc.load_state_dict(model_dict["encoder"])
        for l in optimizerD_glo.param_groups:
            l['lr'] = config.global_disc_lr

        cont_epoch = model_dict["epoch"] + 1

    else:
        cont_epoch = None
    
    print("[INFO] Total Epochs:", config.train_epochs)
    
    # init trainer
    trainer = FaceClozeTrainer(
        model=net,
        model_name=model_name,
        train_loader=tr_data_loader,
        test_loader=val_data_loader,
        epochs=config.train_epochs,
        optimizers={
            "seq_encoder": optimizerSeq,
            "encoder": optimizerEnc,
        },
        grad_clip=config.g_clip,
        save_dir=base_dir + 'playground/face_cloze/',
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
    # cont_model = "playground/ssuper_global_dcgan/ckpts/lstm_ssuper_global_dcgan_model-checkpoint-epoch31.pth"
    cont_model = None
    
    tr_data = GoldenFaceClozeDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path, 
        golden_age_config.annot_path,
        config.panel_size,
        config.img_size,
        4,
        random_order=False,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=True,
        augment=True,
        limit_size=-1)
    
    val_data = GoldenFaceClozeDataset(
        golden_age_config.panel_path,
        golden_age_config.sequence_path, 
        golden_age_config.annot_path,
        config.panel_size,
        config.img_size,
        4,
        random_order=False,
        train_test_ratio=golden_age_config.train_test_ratio,
        train_mode=False,
        augment=False,
        limit_size=-1)
    
    tr_data_loader = DataLoader(tr_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    model_name ="face_cloze_model"
    
    model = train(tr_data_loader, val_data_loader, config, model_name, cont_model)
    torch.save(model, base_dir + 'playground/face_cloze/results/' + model_name + ".pth")