from torch import optim
from torch.utils.data import DataLoader

from data.datasets.ffhq_dataset import FFHQDataset
from data.datasources.ffhq_datasource import FFHQDatasource
from data.datasets.golden_faces import GoldenFacesDataset

from networks.intro_vae import IntroVAE
from training.intro_vae_trainer import IntroVAETrainer
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
    logging.info('Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/intro_vae/results/' + model_name + ".pth")


def continue_training(model_name, train_golden_face=True, cont_epoch=1):
    logging.info("Continuing training...")
    config = read_config(Config.VAE)

    # loading datasets
    if train_golden_face:
        golden_age_config = read_config(Config.GOLDEN_AGE)
        train_dataset = GoldenFacesDataset(
            golden_age_config.faces_path, config.image_dim, limit_size=config.num_training_samples, augment=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_dataset = GoldenFacesDataset(
            golden_age_config.faces_path, config.image_dim, limit_size=config.num_test_samples, augment=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_dataset = FFHQDataset(datasource=FFHQDatasource(config, DataSourceMode.TRAIN))
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
        test_dataset = FFHQDataset(datasource=FFHQDatasource(config, DataSourceMode.TEST))
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    # creating model and training details
    net = IntroVAE(image_size=config.image_dim, channels=config.channels, hdim=config.latent_dim_z).to(ptu.device)

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
                         checkpoint_every_epoch=True,
                         best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))

    epoch, losses = trainer.load_checkpoint(epoch=cont_epoch)

    train_losses, test_losses = trainer.train_epochs(starting_epoch=epoch, losses=losses)

    logging.info("Completing training...")

    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "VAE Losses",
                       base_dir + 'playground/intro_vae/' + f'results/vae_plot.png')
    return net


def train(model_name='test_model', train_golden_face=True):
    # loading config
    logging.info("Initiating training...")
    config = read_config(Config.VAE)

    # loading datasets
    if train_golden_face:
        golden_age_config = read_config(Config.GOLDEN_AGE)
        train_dataset = GoldenFacesDataset(
            golden_age_config.faces_path, config.image_dim, limit_size=config.num_training_samples, augment=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_dataset = GoldenFacesDataset(
            golden_age_config.faces_path, config.image_dim, limit_size=config.num_test_samples, augment=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_dataset = FFHQDataset(datasource=FFHQDatasource(config, DataSourceMode.TRAIN))
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
        test_dataset = FFHQDataset(datasource=FFHQDatasource(config, DataSourceMode.TEST))
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    # creating model and training details
    net = IntroVAE(image_size=config.image_dim, channels=config.channels, hdim=config.latent_dim_z).to(ptu.device)

    test_criterion = elbo

    optimizer_e = optim.Adam(net.encoder.parameters(),
                             lr=config.lr,
                             betas=(config.beta_1, config.beta_2),
                             weight_decay=config.weight_decay)

    optimizer_g = optim.Adam(net.decoder.parameters(),
                             lr=config.lr,
                             betas=(config.beta_1, config.beta_2),
                             weight_decay=config.weight_decay)

    scheduler_e = optim.lr_scheduler.LambdaLR(optimizer_e,
                                              lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                              last_epoch=-1)

    scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g,
                                              lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                              last_epoch=-1)

    # init trainer
    trainer = IntroVAETrainer(model=net,
                              model_name=model_name,
                              test_criterion=test_criterion,
                              train_loader=train_dataloader,
                              test_loader=test_dataloader,
                              epochs=config.train_epochs,
                              optimizer_g=optimizer_g,
                              optimizer_e=optimizer_e,
                              scheduler_e=scheduler_e,
                              scheduler_g=scheduler_g,
                              grad_clip=config.g_clip,
                              checkpoint_every_epoch=True,
                              best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))

    train_losses, test_losses = trainer.train_epochs()

    logging.info("Completing training...")

    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "VAE Losses",
                       base_dir + 'playground/intro_vae/' + f'results/vae_plot.png')
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    cont_epoch = 1

    # continue_training(model_name='10-05-2021-13-49-23_model', train_golden_face=False, cont_epoch=cont_epoch)
    model = train(get_dt_string() + "_model", train_golden_face=False)
    # torch.save(model, base_dir + 'playground/intro_vae/results/' + "test_model.pth")
    # model = torch.load(base_dir + 'playground/intro_vae/results/' + "test_model.pth")
    # model.save_samples(200, base_dir + 'playground/intro_vae/results/end_samples.png' )
    # TODO: add visualizations of reconstruction and sampling from model
