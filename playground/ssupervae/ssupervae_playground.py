from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SSuperVAE

from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.datetime_utils import get_dt_string
from utils.plot_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *
from functional.losses.elbo import elbo


def save_best_loss_model(model_name, model, best_loss):
    print('[INFO] Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/ssupervae/weights/' + model_name + ".pth")


def train(tr_data_loader,
          val_data_loader,
          config,
          model_name='plain_ssupervae',
          cont_model=None):
    # loading config
    print("[INFO] Initiate training...")

    # creating model and training details
    net = SSuperVAE(backbone=config.backbone,
                    latent_dim=config.latent_dim,
                    embed_dim=config.embed_dim,
                    use_lstm=config.use_lstm,
                    seq_size=config.seq_size,
                    enc_channels=config.enc_channels,
                    img_size=config.img_size,
                    lstm_hidden=config.lstm_hidden,
                    lstm_dropout=config.lstm_dropout,
                    fc_hidden_dims=config.fc_hidden_dims,
                    fc_dropout=config.fc_dropout,
                    num_lstm_layers=config.num_lstm_layers,
                    masked_first=config.masked_first)

    criterion = elbo

    if config.parallel:
        net = nn.DataParallel(net).cuda()  # all GPU devices available are used by default
        optimizer = optim.Adam(net.module.parameters(),
                               lr=config.lr,
                               betas=(config.beta_1, config.beta_2),
                               weight_decay=config.weight_decay)
    else:
        net = net.to(ptu.device)
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
                         train_loader=tr_data_loader,
                         test_loader=val_data_loader,
                         epochs=config.train_epochs,
                         optimizer=optimizer,
                         parallel=config.parallel,
                         scheduler=scheduler,
                         grad_clip=config.g_clip,
                         best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l),
                         save_dir=base_dir + 'playground/ssupervae/',
                         checkpoint_every_epoch=True
                         )

    if cont_model is not None:
        epoch, losses = trainer.load_checkpoint(alternative_chkpt_path=cont_model)
        print("[INFO] Continues from loaded model in epoch:", epoch)
        scheduler.step()
    else:
        epoch, losses = None, {}

    train_losses, test_losses = trainer.train_epochs(starting_epoch=epoch, losses=losses)

    print("[INFO] Completed training!")

    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "Plain_SSuperVAE Losses",
                       base_dir + 'playground/supervae/' + f'results/ssupervae_plot.png'
                       )
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    config = read_config(Config.SSUPERVAE)

    # Data Loading

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
        limit_size=1000)

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
        limit_size=100)

    tr_data_loader = DataLoader(tr_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=4)

    print("\nGolden Age Config:", golden_age_config)
    print("\nModel Config:", config)

    if config.use_lstm:
        model_name = "lstm_ssupervae_model" + get_dt_string()
    else:
        model_name = "plain_ssupervae_model" + get_dt_string()

    model = train(tr_data_loader, val_data_loader, config, model_name, cont_model)

    torch.save(model, base_dir + 'playground/ssupervae/results/' + "ssuper_vae_model.pth")
