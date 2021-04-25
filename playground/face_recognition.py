import torchvision.transforms

from data import facedataset, facedatasource
from utils.config_utils import read_config, Config
from utils.image_utils import imshow
from torch.utils.data import Dataset, DataLoader
from utils import pytorch_util as ptu
import torch
from networks.siamese_network import SiameseNetwork
from losses.contrastive_loss import ContrastiveLoss
from torch import optim
from training.face_recognition_trainer import train_epochs
from data.datasource_mode import DataSourceMode
from utils.plot_utils import *
from accuracy.measure_dissimilarity import * 

def compare_test_set(model):
    model.eval()
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(
    datasource=facedatasource.ICartoonFaceDatasource(config, DataSourceMode.TEST))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    compare_image_pairs(iter(dataloader), model)

def visualize_data():
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(
        datasource=facedatasource.ICartoonFaceDatasource(config, DataSourceMode.TRAIN))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    dataiter = iter(dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # TODO: this can be shown better as top bottom row combo
    imshow(torchvision.utils.make_grid(concatenated, nrow=4))
    print(ptu.get_numpy(example_batch[2]))


def save_best_loss_model(model_name, model, best_loss):
    print('current best loss: ' + str(best_loss))
    torch.save(model, model_name + ".pth")
    
    
def train_siamese(model_name='test_model'):
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
                                            best_loss_action = lambda m, l: save_best_loss_model(model_name ,m, l))
    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "Siamese Results",
                       f'results/siamese_train_plot.png')
    return net

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    # model = train_siamese()
    #torch.save(model, "test_model.pth")
    #model = torch.load("test_model.pth")
    #compare_test_set(model)
    visualize_data()