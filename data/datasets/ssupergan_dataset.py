import os
import torch
import pandas as pd
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL.Image import Image as PilImage
import textwrap, os
import re 
import pickle
import numpy as np
import statistics
import copy
import torchvision.transforms.functional as FT


from data.datasets.ssupergan_preprocess import *


def read_images(file_names, folder_path):
    """
    Takes file names and their folder path as input
    Return List of Images
    """
    paths = [os.path.join(folder_path,file) for file in file_names ]
    
    return [Image.open(path) for path in paths]



class SSGANDataset(Dataset):
    """ SSGAN Dataset """
    def __init__(self, annotations, face_annotations, root_dir, transform, face_transforms):
        """
        Input Annotation File --> [index, ]
        """
        #self.annotations = pd.read_json(annotation_file)
        #print("CWD Dataset", os.getcwd())
        self.annotations = annotations
        self.face_annotations = face_annotations
        self.root_dir = root_dir
        self.transform = transform
        self.face_transforms = face_transforms
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,idx):
        #print("İndex ",idx)
        # Returns Folder of the sequence
        folder = self.annotations.iloc[idx].folder
        # List of Sequences
        sequences = self.annotations.iloc[idx].sequence
        
        
        
        sequence_path = os.path.join(self.root_dir,folder)
        #print("Folder", folder, "Sequence ",sequences, "Sequence path ",sequence_path)
        #PILLOW 
        images = read_images(sequences, sequence_path)
    

        #Name of the image in the last sequence
        the_last_sequence = sequences[-1]
        #Read Face Annotations of Last Panel
        last_panel_face_annotations = self.face_annotations[folder][the_last_sequence]

        # Select Random Face, In the preprocessing part all the faces smaller than 32*32 eliminated
        selected_face = np.random.randint(len(last_panel_face_annotations))
        
        
        last_image = np.array(images[-1])
        
        H_LAST, W_LAST, C_LAST = last_image.shape
        
        #print("Last Original Image Shape ", last_image.shape)
        
        # Face that will be masked 
        face_annotation_read = last_panel_face_annotations[selected_face]#.annotation
        face_annotation = face_annotation_read
        
        #print("ILK OKUDU ANNOTATION ",face_annotation)
        
        #Xmin
        face_annotation[0] = max(face_annotation[0].item(),0)
        #Ymin
        face_annotation[1] = max(face_annotation[1].item(),0)
        #Xmax
        face_annotation[2] = min(face_annotation[2].item(),W_LAST)
        #ymax
        face_annotation[3] = min(face_annotation[3].item() ,H_LAST)
        
        #print("Normalized ANnotation " ,face_annotation )
        
        
        # Last Image Should be crop in more detail
        # Make it square
        
        
        last_image_2, face_annotation_2 = crop_last_image(last_image, face_annotation)
        
        #print("AFTER CROP LAST IMAGE 2 ",face_annotation_2)
        #crop_face # Put Border Around It
        
        face, face_annotation_2 =  crop_face_v2(last_image_2, face_annotation_2)
        
        #print(face_annotation_2)
        x_min_face, y_min_face, x_max_face, y_max_face  = face_annotation_2
        

        last_image_2[int(y_min_face) :int(y_max_face), int(x_min_face): int(x_max_face),:] = 0
        
        #print("YYey")
    
        #TODO Performance
        last_image_2 = Image.fromarray(last_image_2)
        last_image_2, face_annotation_2 = resize(last_image_2, face_annotation_2)
        
        #print("RESIZE OUTPUT LAST IMAGE 2 ",last_image_2 , " face_annotation_2 ",face_annotation_2)
        
        
        # Fce is er
        try:
            face = Image.fromarray(face)
        except:
            print("Problem with Face ",face.shape, face)
        
        
        #
        result = []
        if self.transform:
            for each in images[:-1]:
                result.append(torch.unsqueeze(self.transform(each), dim=0))

            result.append(torch.unsqueeze(transforms.ToTensor()(last_image_2), dim=0))
            #face = torch.unsqueeze(self.face_transforms(face),0)
            face = self.face_transforms(face)
            

        
        
        sample = {'image':torch.cat(result,dim=0) , "target": face}
        return sample
        
        #return last_image,face_annotation, last_image_2,face_annotation_2,last_panel_face_annotations
        

            
    
def create_dataset_SSGAN(sequence_annotations, face_annotations, center_crop_dimension = (400,400), resize_dimension = (300,300), face_resize = (64,64), FOLDER_PATH = "/datasets/COMICS/raw_panel_images", annotation_file = "annotations2.json"):
    
    
    panel_transforms = transforms.Compose(
        [
        transforms.CenterCrop(center_crop_dimension),
        transforms.Resize(resize_dimension),
        transforms.ToTensor()
        ]
    )
    
    face_transforms = transforms.Compose(
        [
        transforms.Resize(face_resize),
        transforms.ToTensor()
        ]
    )
    
    dataset = SSGANDataset(sequence_annotations, face_annotations, FOLDER_PATH, panel_transforms,face_transforms)
    
    return dataset


def create_data_loader(dataset, batchsize = 32, shuffle=True, split = "train", num_workers = 0):
    dataloader = DataLoader(dataset, batch_size=batchsize,
                        shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
    
    
        
        

        
        