import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from data.facedatasource import FaceDatasource
from utils.pytorch_util import from_numpy

class FaceDataset(Dataset):
    def __init__(self,
                 datasource: FaceDatasource,
                 transformations: transforms = None):
        if transformations is None:
            transformations = []
        transformations.insert(0, transforms.ToTensor())
        self.transforms = transforms.Compose(transformations)
        self.datasource = datasource

    def __getitem__(self, index):
        data, face_id = self.datasource.get_item(index)
        data = self.transforms(data)
        # create labels from face_id
        return (data, face_id)  # (img, label)

    def __len__(self):
        return self.datasource.compute_length()
