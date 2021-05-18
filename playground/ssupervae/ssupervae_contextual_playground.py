import os
import sys
import json
from copy import deepcopy

from torch import optim
from torch.utils.data import DataLoader

from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_panels import GoldenPanelsDataset

from networks.plain_ssupervae import PlainSSuperVAE
from networks.ssupervae_contextual_attentional import SSuperVAEContextualAttentional
from training.ssupervae_contextual_attn_trainer import SSuperVAEContextualAttentionalTrainer
from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *
from utils import pytorch_util as ptu
from utils.image_utils import *
from configs.base_config import *
from functional.losses.elbo import *


def save_best_loss_model(model_name, model, best_loss):
    print('[INFO] Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/ssupervae/weights/' + model_name + ".pth")


def train(data_loader,
          config,
          panel_dim,
          elbo_criterion,
          model_name='plain_ssupervae',
          cont_epoch=-1,
          cont_model=None):
    # loading config
    print("[INFO] Initiate training...")

    # creating model and training details
    net = SSuperVAEContextualAttentional(config.backbone,
                                         panel_img_size=panel_dim,
                                         latent_dim=config.latent_dim,
                                         embed_dim=config.embed_dim,
                                         seq_size=config.seq_size,
                                         decoder_channels=config.decoder_channels,
                                         gen_img_size=config.image_dim).to(ptu.device)

    criterion = elbo_criterion

    optimizer = optim.Adam(net.parameters(),
                           lr=config.lr,
                           betas=(config.beta_1, config.beta_2),
                           weight_decay=config.weight_decay)

    d_params = list(net.local_disc.parameters()) + list(net.global_disc.parameters())
    optimizer_disc = optim.Adam(d_params,
                                lr=config.lr,
                                betas=(config.beta_1, config.beta_2),
                                weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                            last_epoch=-1)

    scheduler_disc = optim.lr_scheduler.LambdaLR(optimizer_disc,
                                                 lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                 last_epoch=-1)
    # init trainer
    trainer = SSuperVAEContextualAttentionalTrainer(model=net,
                                                    config_disc=config,
                                                    model_name=model_name,
                                                    criterion=criterion,
                                                    train_loader=data_loader,
                                                    test_loader=None,
                                                    epochs=config.train_epochs,
                                                    optimizer=optimizer,
                                                    optimizer_disc=optimizer_disc,
                                                    scheduler=scheduler,
                                                    scheduler_disc=scheduler_disc,
                                                    grad_clip=config.g_clip,
                                                    best_loss_action=lambda m, l: save_best_loss_model(model_name, m,
                                                                                                       l),
                                                    save_dir=base_dir + 'playground/ssupervae/',
                                                    checkpoint_every_epoch=True
                                                    )

    if cont_epoch > -1:
        epoch, losses = trainer.load_checkpoint(epoch=cont_epoch)
    elif cont_model is not None:
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


def test_forward_for_ssupervae_context():
    ptu.set_gpu_mode(True)
    config = read_config(Config.PLAIN_SSUPERVAE)
    golden_age_config = read_config(Config.GOLDEN_AGE)

    panel_dim = golden_age_config.panel_dim[0]

    cont_epoch = -1
    cont_model = None  # "playground/ssupervae/weights/model-18.pth"
    limit_size = 32

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
                               limit_size=limit_size)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    # creating model and training details
    net = SSuperVAEContextualAttentional(config.backbone,
                                         panel_img_size=panel_dim,
                                         latent_dim=config.latent_dim,
                                         embed_dim=config.embed_dim,
                                         seq_size=config.seq_size,
                                         decoder_channels=config.decoder_channels,
                                         gen_img_size=config.image_dim).to(ptu.device)

    dataiter = iter(data_loader)
    x, y, mask, mask_coordinates = next(dataiter)

    x = x.to(ptu.device)
    y = y.to(ptu.device)
    mask = mask.to(ptu.device)

    z, _, mu_z, mu_x, logstd_z = net(x)

    # Preparing for Fine Generator
    B, S, C, W, H = x.shape
    mask = mask.view(B, 1, W, H)
    x_stage_0 = ptu.zeros(B, C, H, W)
    x_stage_1 = ptu.zeros_like(x_stage_0)
    for i in range(len(x)):
        last_panel = x[i, 2, :, :, :]
        output_merged_last_panel = deepcopy(last_panel)

        last_panel_face = y[i, :, :, :]
        last_panel_output_face = mu_x[0, :, :, :]

        mask_coordinates_n = ptu.get_numpy(mask_coordinates[i])

        original_w = abs(mask_coordinates_n[0] - mask_coordinates_n[1])
        original_h = abs(mask_coordinates_n[2] - mask_coordinates_n[3])

        # inserting original face to last panel
        modified = last_panel_face.view(1, *last_panel_face.size())
        interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                             size=(original_w, original_h))
        interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
        last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
        mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
        x_stage_0[i, :, :, :] = last_panel

        # inserting output face to last panel
        modified = last_panel_output_face.view(1, *last_panel_output_face.size())
        interpolated_last_panel_face_batch = torch.nn.functional.interpolate(modified,
                                                                             size=(original_w, original_h))
        interpolated_last_panel_face = interpolated_last_panel_face_batch[0]
        output_merged_last_panel[:, mask_coordinates_n[0]: mask_coordinates_n[1],
        mask_coordinates_n[2]: mask_coordinates_n[3]] = interpolated_last_panel_face
        x_stage_1[i, :, :, :] = output_merged_last_panel

    # TODO: x_stage_2 here is not in same normalized space i assume
    # we should cross check with the implementation
    # TODO: we need to normalize before visualizing
    x_stage_2, offset_flow = net.fine_generator(x_stage_0, x_stage_1, mask)

    print("here")


if __name__ == '__main__':
    # test_forward_for_ssupervae_context()

    ptu.set_gpu_mode(True)
    config = read_config(Config.VAE_CONTEXT_ATTN)
    golden_age_config = read_config(Config.GOLDEN_AGE)

    panel_dim = golden_age_config.panel_dim[0]

    cont_epoch = -1
    cont_model = None  # "playground/ssupervae/weights/model-18.pth"
    limit_size = 3000

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
                               limit_size=limit_size)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = train(data_loader,
                  config,
                  model_name=get_dt_string() + "_model",
                  cont_epoch=cont_epoch,
                  cont_model=cont_model,
                  panel_dim=panel_dim,
                  elbo_criterion=elbo_with_L1)
    torch.save(model, base_dir + 'playground/ssupervae/results/' + "ssuper_vae_model.pth")

    """
    """
