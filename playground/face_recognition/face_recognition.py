import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms

from utils.config_utils import read_config, Config
from utils.image_utils import imshow
from utils.logging_utils import *
from utils.datetime_utils import get_dt_string
from utils.plot_utils import *
from utils import pytorch_util as ptu

from data import facedataset, facedatasource
from data.datasource_mode import DataSourceMode
from networks.siamese_network import SiameseNetwork
from functional.losses.contrastive_loss import ContrastiveLoss
from functional.metrics.dissimilarity import *
from training.face_recognition_trainer import train_epochs
from configs.base_config import *


def compare_test_set(model, max_display=None):
    model.eval()
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, DataSourceMode.TEST))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    compare_image_pairs(iter(dataloader), model, max_display=max_display)


def visualize_data():
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, DataSourceMode.TRAIN))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    dataiter = iter(dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    #  TODO: this can be shown better as top bottom row combo
    imshow(torchvision.utils.make_grid(concatenated, nrow=4))
    print(ptu.get_numpy(example_batch[2]))


def save_best_loss_model(model_name, model, best_loss):
    print('current best loss: ' + str(best_loss))
    logging.info('current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/face_recognition/results/' + model_name + ".pth")


def train_siamese(model_name='test_model'):
    logging.info("initiate training")
    config = read_config(Config.FACE_RECOGNITION)
    train_dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, mode=DataSourceMode.TRAIN))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, mode=DataSourceMode.TEST))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    net = SiameseNetwork(image_dim=config.image_dim).to(ptu.device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    train_losses, test_losses = train_epochs(net,
                                             optimizer,
                                             criterion,
                                             train_loader=train_dataloader,
                                             test_loader=test_dataloader,
                                             train_args=dict(epochs=config.train_epochs),
                                             best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))
    logging.info("completed training")
    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "Siamese Results",
                       base_dir + 'playground/face_recognition/' + f'results/siamese_train_plot.png')
    return net


def compute_mean_acc(model, datasource_mode: DataSourceMode = DataSourceMode.TEST):
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, mode=datasource_mode))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    acc = compute_mean_accuracy(dataiterator=dataloader,
                                model=model)
    print("Accuracy: " + str(acc))


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    model = train_siamese(get_dt_string() + "_model")
    torch.save(model, base_dir + 'playground/face_recognition/results/' + "test_model.pth")
    # model = torch.load("test_model.pth")
    # compare_test_set(model)
    # visualize_data()
    # compute_mean_acc(datasource_mode=DataSourceMode.TRAIN)
