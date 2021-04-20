import torchvision.transforms

from data import facedataset, facedatasource
from utils.config_utils import read_config, Config
from torch.utils.data import Dataset, DataLoader
from utils import pytorch_util as ptu

if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    config = read_config(Config.FACE_RECOGNITION)
    dataset = facedataset.FaceDataset(datasource=facedatasource.ICartoonFaceDatasource(config))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch)