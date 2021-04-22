import torchvision.transforms

from data import facedataset, facedatasource
from utils.config_utils import read_config, Config
from utils.image_utils import imshow
from torch.utils.data import Dataset, DataLoader
from utils import pytorch_util as ptu
import torch


def visualize_data():
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.PairedFaceDataset(datasource=facedatasource.ICartoonFaceDatasource(config))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    dataiter = iter(dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated, nrow=2))
    print(ptu.get_numpy(example_batch[2]))


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    visualize_data()
