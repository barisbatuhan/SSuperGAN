from torch import optim
from torch.utils.data import DataLoader

import os
import sys
os.path.dirname(sys.executable)
sys.path.append("/kuacc/users/baristopal20/SSuperGAN/")

from data.datasets.random_dataset import RandomDataset
from training.ssupergan_trainer import SSuperGANTrainer
from networks.ssupergan import SSuperGAN
from utils.config_utils import read_config, Config
from utils import pytorch_util as ptu
from utils.logging_utils import *
from utils.plot_utils import *
from configs.base_config import *


def train_model(config, data_loader, save_dir=None):
    logging.info("[INFO] SSuperGAN training is starting...")
    
    seq_enc_args = {
        "lstm_hidden": config.lstm_hidden,
        "embed": config.latent_dim,
        "cnn_embed": config.cnn_embed,
        "fc_hiddens": config.fc_hiddens,
        "lstm_dropout": config.lstm_dropout,
        "fc_dropout": config.fc_dropout,
        "num_lstm_layers": config.num_lstm_layers
    }
    
    net = SSuperGAN( 
        seq_enc_args,
        image_dim=config.image_dim,
        latent_dim=config.latent_dim,
        g_hidden_size=config.g_hidden_size,
        d_hidden_size=config.d_hidden_size,
        e_hidden_size=config.e_hidden_size, 
        gan_type="bigan"
    ).to(ptu.device)

    d_optimizer = torch.optim.Adam(net.gan.discriminator.parameters(),
                                   lr=config.discriminator_lr,
                                   betas=(config.discriminator_beta_1, config.discriminator_beta_2),
                                   weight_decay=config.discriminator_weight_decay)

    g_optimizer = torch.optim.Adam(list(net.gan.encoder.parameters()) + list(net.gan.generator.parameters()),
                                   lr=config.generator_lr,
                                   betas=(config.generator_beta_1, config.generator_beta_2),
                                   weight_decay=config.generator_weight_decay)
    
    seq_enc_optimizer = torch.optim.Adam(net.seq_encoder.parameters(),
                                   lr=config.seq_encoder_lr,
                                   betas=(config.seq_encoder_beta_1, config.seq_encoder_beta_2),
                                   weight_decay=config.seq_encoder_weight_decay)
    
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    
    seq_enc_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
                                                          lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                          last_epoch=-1)
    
    trainer = SSuperGANTrainer(net,
                               data_loader,
                               config.train_epochs,
                               d_optimizer,
                               g_optimizer,
                               seq_enc_optimizer,
                               scheduler_disc=d_scheduler,
                               scheduler_gen=g_scheduler,
                               scheduler_seq_enc=seq_enc_scheduler,
                               gen_steps=1,
                               disc_steps=1,
                               save_dir=save_dir)
    
    
    losses = trainer.train_model()

    logging.info("[INFO] Completed training!")
    save_training_plot(losses['discriminator_loss'],
                       losses['generator_loss'],
                       "SSuperGAN Losses",
                       save_dir + f'results/ssupergan_plot.png')
    return net


if __name__ == '__main__':
    config = read_config(Config.SSUPERGAN)
    save_dir = "playground/ssupergan/"
    data = RandomDataset((3, 3, 360, 360), (3, config.image_dim, config.image_dim))
    data_loader = DataLoader(data, batch_size=config.batch_size)
    model = train_model(config, data_loader, save_dir=save_dir)
