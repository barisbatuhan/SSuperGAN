from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_panels import GoldenPanelsDataset
from networks.panel_encoder.cnn_embedder import CNNEmbedder
from networks.sort_sequence_network import SortSequenceNetwork

from networks.ssupervae import SSuperVAE
from training.sort_sequence_trainer import SortSequenceTrainer
from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *
from functional.losses.elbo import elbo


def save_best_loss_model(model_name, model, best_loss):
    print('[INFO] Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/sort_sequence/weights/' + model_name + ".pth")


def train(data_loader,
          test_dataloader,
          panel_dim,
          config,
          model_name='sort_sequence',
          cont_epoch=-1,
          cont_model=None):
    # loading config
    print("[INFO] Initiate training...")
    cnn_embedder = CNNEmbedder("efficientnet-b5", embed_dim=config.embed_dim)
    net = SortSequenceNetwork(embedder=cnn_embedder,
                              num_elements_in_sequence=config.seq_size,
                              pairwise_extraction_in_size=(panel_dim ** 2) * 4).cuda()

    if config.parallel:
        net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(),
                           lr=config.lr,
                           betas=(config.beta_1, config.beta_2),
                           weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                            last_epoch=-1)

    # init trainer
    # Criterion is None because network itself contains loss function
    trainer = SortSequenceTrainer(model=net,
                                  model_name=model_name,
                                  criterion=None,
                                  train_loader=data_loader,
                                  test_loader=test_dataloader,
                                  epochs=config.train_epochs,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  grad_clip=config.g_clip,
                                  best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l),
                                  save_dir=base_dir + 'playground/sort_sequence/',
                                  checkpoint_every_epoch=True)

    if cont_epoch > -1:
        epoch, losses = trainer.load_checkpoint(epoch=cont_epoch)
    elif cont_model is not None:
        epoch, losses = trainer.load_checkpoint(alternative_chkpt_path=cont_model)
        print("[INFO] Continues from loaded model in epoch:", epoch)
        scheduler.step()
    else:
        epoch, losses = None, {}

    initiate_logger()
    train_losses, test_losses = trainer.train_epochs(starting_epoch=epoch, losses=losses)

    print("[INFO] Completed training!")

    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "SORT SEQUENCE NETWORK LOSSES",
                       base_dir + 'playground/sort_sequence/' + f'results/sort_sequence_plot.png'
                       )
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    config = read_config(Config.SORT_SEQUENCE)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    cont_epoch = -1
    cont_model = None  # "playground/ssupervae/weights/model-18.pth"
    limit_size = -1
    data = GoldenPanelsDataset(golden_age_config.panel_path,
                               golden_age_config.sequence_path,
                               golden_age_config.panel_dim,
                               config.image_dim,
                               augment=False,
                               mask_val=False,
                               mask_all=False,
                               return_mask=False,
                               return_gt_last_panel=True,
                               train_test_ratio=golden_age_config.train_test_ratio,
                               train_mode=True,
                               limit_size=limit_size)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=4)

    test_data = GoldenPanelsDataset(golden_age_config.panel_path,
                                    golden_age_config.sequence_path,
                                    golden_age_config.panel_dim,
                                    config.image_dim,
                                    augment=False,
                                    mask_val=golden_age_config.mask_val,
                                    mask_all=golden_age_config.mask_all,
                                    return_mask=golden_age_config.return_mask,
                                    return_gt_last_panel=True,
                                    train_test_ratio=golden_age_config.train_test_ratio,
                                    train_mode=False,
                                    limit_size=limit_size)
    test_data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=4)

    model_name = 'sort_sequence_' + get_dt_string()

    model = train(data_loader,
                  test_data_loader,
                  golden_age_config.panel_dim[0],
                  config,
                  model_name,
                  cont_epoch=cont_epoch,
                  cont_model=cont_model)

    torch.save(model, base_dir + 'playground/sort_sequence/results/' + model_name + ".pth")
