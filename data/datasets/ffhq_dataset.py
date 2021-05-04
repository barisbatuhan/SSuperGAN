import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.datasources.base_datasource import BaseDatasource
import random
from utils import pytorch_util as ptu


class FFHQDataset(Dataset):
    def __init__(self,
                 datasource: BaseDatasource,
                 transformations: transforms = None):
        random.seed(10)
        if transformations is None:
            transformations = []
        transformations.insert(0, transforms.ToTensor())
        transformations.append(transforms.Lambda(lambda x: x.to(device=ptu.device, dtype=torch.float32)))
        self.transforms = transforms.Compose(transformations)
        self.datasource = datasource

    def __getitem__(self, index):
        data = self.datasource.get_item(index)
        data = self.transforms(data)
        return data

    def __len__(self):
        return self.datasource.compute_length()
