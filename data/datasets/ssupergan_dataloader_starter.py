import os
import sys

os.path.dirname(sys.executable)
sys.path.append("/scratch/users/baristopal20/SSuperGAN/")

from data.datasets.ssupergan_preprocess import *
from data.datasets.ssupergan_dataset import *

dataset_path = "/datasets/COMICS"
golden_annotations_path="/userfiles/comics_grp/golden_age/face_annots"

# 
folder_structure = return_folder_structure()
face_annotations,num_small_faces = read_face_detection_annotations(golden_annotations_path)
all_possible_panel_sequence_face_detection = return_all_possible_panel_seq_face_detection(face_annotations)
valid_sequences = valid_sequence_creator_from_face_annotations(all_possible_panel_sequence_face_detection, folder_structure)
annot_file = create_annotation_file(valid_sequences)
annotations = pd.DataFrame(annot_file)
annotations.to_json("golden_panel_annots.json")
d2 = create_dataset_SSGAN(face_annotations)

dataloader = create_data_loader(d2)


"""
for x in dataloader:
    print(x)
    break


"""