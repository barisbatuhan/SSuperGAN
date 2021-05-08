import os
import sys


import os
import sys

os.path.dirname(sys.executable)
sys.path.append("/scratch/users/ckoksal20/COMP547Project/AF-GAN/")

from data.datasets.ssupergan_preprocess import *
from data.datasets.ssupergan_dataset import *

dataset_path = "/datasets/COMICS"
golden_annotations_path="/kuacc/users/ckoksal20/COMP547Project/golden_annot"



folder_structure = return_folder_structure()
face_annotations,num_small_faces = read_face_detection_annotations(golden_annotations_path)

all_possible_panel_sequence_face_detection = return_all_possible_panel_seq_face_detection(face_annotations)
valid_sequences = valid_sequence_creator_from_face_annotations(all_possible_panel_sequence_face_detection, folder_structure)

annotations = create_annotation_file(valid_sequences)

#annotations.to_json("annotations2.json")

dataset = create_dataset_SSGAN(annotations, face_annotations)

dataloader = create_data_loader(dataset)

"""
data = 0
for i,x in enumerate(dataloader):
    data = x
    break
    
"""


