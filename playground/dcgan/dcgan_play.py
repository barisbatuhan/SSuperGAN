import os
import sys
import json
os.path.dirname(sys.executable)
sys.path.append("/home/ckoksal20/Desktop/COMP547/SSuperGAN/")


from torch import optim
from torch.utils.data import DataLoader

from data.datasets.random_dataset import RandomDataset
from data.datasets.ssupergan_dataset import SSGANDataset
from data.datasets.golden_faces import GoldenFacesDataset
from data.datasets.ssupergan_preprocess import *
from data.datasets.ssupergan_dataset import *


from training.dcgan_trainer import DCGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

from torchvision.datasets import ImageFolder
from networks.dcgan import DCGAN


def save_best_loss_model(model_name, model, best_loss):
    # print('current best loss: ' + str(best_loss))
    logging.info('Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/dcgan/weights/' + model_name + ".pth")


def train(data_loader,config, model_name='dcgan'):
    # loading config
    logging.info("[INFO] Initiate training...")

    # creating model and training details

    nc = config.nc
    nz = config.nz
    ngf = config.ngf
    ndf = config.ndf
    beta1 = config.beta1
    lr = config.lr
    epochs = config.num_epochs
    ngpu = config.ngpu
    image_size = config.image_size

    net = DCGAN(ngpu, image_size, nc, nz, ngf, ndf)
    


    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(net.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(net.generator.parameters(), lr=lr, betas=(beta1, 0.999))


    # init trainer

    trainer = DCGANTrainer(model = net,
                           data_loader = data_loader,
                           epochs = 10,
                           optimizer_disc = optimizerD,
                           optimizer_gen = optimizerG,
                           save_dir=base_dir + 'playground/dcgan/'

    )
    
    losses = trainer.train_epochs()

    logging.info("[INFO] Completed training!")
    
    save_training_plot(losses['gen'],
                       losses['disc'],
                       "DCGAN Faces Losses",
                       base_dir + 'playground/dcgan/' + f'results/dcgan_plot.png'
                      )
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)

    config = read_config(Config.DCGAN)
    #print("type config ", type(config), " CONFIG : : ", config)
    image_size = config.image_size
    batch_size  = config.batch_size
    workers = config.workers

    dataset_path = config.dataset_path

    dataset = ImageFolder(root=dataset_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
 

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


    
    model = train(data_loader, config, get_dt_string() + "_model")
    torch.save(model, base_dir + 'playground/dcgan/results/' + "dcgan_model.pth")
