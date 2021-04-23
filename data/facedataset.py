import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.facedatasource import FaceDatasource
from utils.pytorch_util import from_numpy
import random
from utils import pytorch_util as ptu


class FaceDataset(Dataset):
    def __init__(self,
                 datasource: FaceDatasource,
                 transformations: transforms = None):
        if transformations is None:
            transformations = []
        transformations.insert(0, transforms.ToTensor())
        transformations.append(transforms.Lambda(lambda x: x.to(ptu.device)))
        self.transforms = transforms.Compose(transformations)
        self.datasource = datasource

    def __getitem__(self, index):
        data, face_id = self.datasource.get_item(index)
        data = self.transforms(data)
        # create labels from face_id
        return (data, face_id)  # (img, label)

    def __len__(self):
        return self.datasource.compute_length()


# Source: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
class PairedFaceDataset(Dataset):
    def __init__(self,
                 datasource: FaceDatasource,
                 transformations: transforms = None):
        if transformations is None:
            transformations = []
        transformations.insert(0, transforms.ToTensor())
        transformations.append(transforms.Lambda(lambda x: x.to(device=ptu.device, dtype=torch.float32)))
        self.transforms = transforms.Compose(transformations)
        self.datasource = datasource

    def __getitem__(self, index):
        img0_tuple_idx = random.choice(range(self.__len__()))
        img0, img0_id = self.datasource.get_item(img0_tuple_idx)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        # Optimization for batching speed

        if should_get_same_class:
            face_data_item = random.choice(self.datasource.data_by_id[img0_id])
            img1, img1_id = self.datasource.data_item_to_actual_data(face_data_item)
        else:
            while True:
                img1_tuple_idx = random.choice(range(self.__len__()))
                img1_id = self.datasource.get_item_id(img1_tuple_idx)
                if img0_id != img1_id:
                    img1, img1_id = self.datasource.get_item(img1_tuple_idx)
                    break
        img0 = self.transforms(img0)
        img1 = self.transforms(img1)
        label = ptu.zeros(1) if should_get_same_class else ptu.ones(1)
        return img0, img1, label

    def __len__(self):
        return self.datasource.compute_length()
