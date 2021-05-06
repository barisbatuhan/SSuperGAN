from torch import optim
from torch.utils.data import DataLoader

from data.datasets.ffhq_dataset import FFHQDataset
from data.datasources.ffhq_datasource import FFHQDatasource
from data.datasources.golden_age_face_datasource import GoldenAgeFaceDatasource
from networks.intro_vae import IntroVAE
from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *

from data.datasets import facedataset
from data.datasources import facedatasource
from data.datasources.datasource_mode import DataSourceMode
from functional.metrics.dissimilarity import *
from training.face_recognition_trainer import train_epochs
from configs.base_config import *
from functional.losses.elbo import elbo


def save_best_loss_model(model_name, model, best_loss):
    # print('current best loss: ' + str(best_loss))
    logging.info('current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/vae/results/' + model_name + ".pth")


def train(model_name='test_model'):
    # loading config
    logging.info("initiate training")

    config = read_config(Config.VAE)

    # loading datasets
    golden_age_config = read_config(Config.GOLDEN_AGE_FACE)
    train_dataset = FFHQDataset(
        datasource=GoldenAgeFaceDatasource(golden_age_config, mode=DataSourceMode.TRAIN))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = FFHQDataset(
        datasource=GoldenAgeFaceDatasource(golden_age_config, mode=DataSourceMode.TEST))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # creating model and training details
    net = IntroVAE(image_size=golden_age_config.image_dim,
                   channels=config.channels,
                   hdim=config.latent_dim_z).to(ptu.device)

    criterion = elbo

    optimizer = optim.Adam(net.parameters(),
                           lr=config.lr,
                           betas=(config.beta_1, config.beta_2),
                           weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                            last_epoch=-1)

    # init trainer
    trainer = VAETrainer(model=net,
                         model_name=model_name,
                         criterion=criterion,
                         train_loader=train_dataloader,
                         test_loader=test_dataloader,
                         epochs=config.train_epochs,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         grad_clip=config.g_clip,
                         best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))
    train_losses, test_losses = trainer.train_epochs()

    logging.info("completed training")
    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "BiGAN Losses",
                       base_dir + 'playground/bigan/' + f'results/bigan_plot.png')
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    model = train(get_dt_string() + "_model")
    # model = train_golden_age_face_bigan(get_dt_string() + "_model")
    torch.save(model, base_dir + 'playground/vae/results/' + "test_model.pth")
    # model = torch.load("test_model.pth")
    # TODO: add visualizations of reconstruction and sampling from model
