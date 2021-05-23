import os
import sys
import json
os.path.dirname(sys.executable)
sys.path.append("/kuacc/users/ckoksal20/COMP547Project/SSuperGAN/")


from torch import optim
from torch.utils.data import DataLoader
from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_faces import GoldenFacesDataset
from data.datasets.golden_panels import GoldenPanelsDataset

from training.dcgan_trainer import DCGANTrainer
from training.ssuper_dcgan_trainer import SSuperDCGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

from networks.ssuper_dcgan import SSuperDCGAN

import torch.nn as nn
import torch.nn as nn




def save_best_loss_model(model_name, model, best_loss):
    # print('current best loss: ' + str(best_loss))
    logging.info('Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/ssuper_dcgan/weights/' + model_name + ".pth")


def train(data_loader,config,dataset, model_name='ssuper_dcgan',):
    # loading config
    logging.info("[INFO] Initiate training...")

    # creating model and training details

    net = SSuperDCGAN(config.backbone, 
                    latent_dim=config.latent_dim, 
                    embed_dim=config.embed_dim,
                    use_lstm=config.use_lstm,
                    seq_size=config.seq_size,
                    gen_img_size=config.image_dim,
                    lstm_hidden=config.lstm_hidden,
                    lstm_dropout=config.lstm_dropout,
                    fc_hidden_dims=config.fc_hidden_dims,
                    fc_dropout=config.fc_dropout,
                    num_lstm_layers=config.num_lstm_layers,
                    masked_first=config.masked_first,
                    ngpu = config.ngpu,
                    ngf = config.ngf,
                    ndf = config.ndf,
                    nc = config.nc,
                    image_size=config.image_dim).to(ptu.device) 
    
    #print(net)
    



    # Setup Adam optimizers for both G and D

    optimizer_encoder = optim.Adam(net.encoder.parameters(), lr=config.lr)

    optimizerD = optim.Adam(net.dcgan.discriminator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2),)
    optimizerG = optim.Adam(net.dcgan.generator.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))


    print("Total epochs ",config.train_epochs)
    
    # init trainer
    trainer = SSuperDCGANTrainer(model=net,
                         model_name=model_name,
                         criterion=nn.BCELoss,
                         train_loader=data_loader,
                         test_loader=None,
                         epochs=config.train_epochs,
                         optimizer_encoder=optimizer_encoder,
                         optimizer_generator= optimizerG,
                         optimized_discriminator= optimizerD,
                         grad_clip=config.g_clip,
                         best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l),
                         save_dir=base_dir + 'playground/ssuper_dcgan/',
                         checkpoint_every_epoch=True
                        )



    losses, test_losses = trainer.train_epochs()

    logging.info("[INFO] Completed training!")
    
    #print("Losses ",losses)
    save_training_plot(losses['gen_loss'],
                       losses['disc_loss'],
                       "SSPUPERGAN  Losses",
                       base_dir + 'playground/ssuper_dcgan/' + f'results/{model_name}_plot.png'
                      )
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    
    config = read_config(Config.SSUPERDCGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    cont_epoch = -1
    cont_model = None # "playground/ssupervae/weights/model-18.pth"
    
    # data = RandomDataset((3, 3, 360, 360), (3, config.image_dim, config.image_dim))
    data = GoldenPanelsDataset(golden_age_config.panel_path,
                               golden_age_config.sequence_path, 
                               golden_age_config.panel_dim,
                               config.image_dim, 
                               augment=False, 
                               mask_val=golden_age_config.mask_val,
                               mask_all=golden_age_config.mask_all,
                               return_mask=golden_age_config.return_mask,
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
        
        
    """model = train(train_dataloader,  config, args.dataset, get_dt_string() + "dcgan_faces_model")
    torch.save(model, base_dir + 'playground/dcgan/results/' + "dcgan_gfaces.pth")"""
        
        