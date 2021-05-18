import os
import re
import sys
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

os.path.dirname(sys.executable)
sys.path.append("/scratch/users/baristopal20/SSuperGAN/")

def read_golden_annots(annot_dir, conf_thold, face_thold, max_classes=3959):
    print("[INFO] Reading Face Annotations...")
    annots = {}
    for i in tqdm(range(max_classes)):
        # read the annot file
        file_path = os.path.join(annot_dir, str(i) + ".txt")
        if not os.path.exists(file_path):
            continue
        
        annots[str(i)] = {}
        
        lines = open(file_path, "r").readlines()
        for line in lines:
            if len(line) < 2:
                # when there is only newline character
                continue
            
            # read the line annots
            fname, x1, y1, x2, y2, conf = line[:-1].split(" ")
            fname = fname.split("/")[1]
            x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf)
     
            if conf < conf_thold:
                continue  
            elif max(x2-x1, y2-y1) < face_thold:
                continue
            
            if fname not in annots:
                annots[str(i)][fname] = []
            annots[str(i)][fname].append([x1, y1, x2, y2])
            
    return annots


def sort_alphanumerical(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key=alphanum_key)


def return_folder_structure(dataset_path="/datasets/COMICS/raw_panel_images/"):
    """
    Input : Dataset_path 
    Output : Dictionary that stores folders as key and images as value
    Reason : To have datastructure to check whether given file_name in the folder or not
    """
    print("[INFO] Reading the Folder Structure...")

    folder_structure = {}
    for folder in tqdm(sort_alphanumerical(os.listdir(dataset_path))):
        folder_path = os.path.join(dataset_path, folder)
        folder_structure[folder] = sort_alphanumerical(os.listdir(folder_path))
        
    return folder_structure

def get_sequential_panels(annots, sorted_dirs, window_size=3):
    """ Returns sequential panels of window size: [[p1, p2, p3], [p2, p3, p4], ...]"""
    print("[INFO] Constructing Sequential Panels...")
    
    seq_panels = []
    for k in tqdm(annots.keys()):
        # sorting the found image files in alphanumerical order
        annot_files = sort_alphanumerical([ *annots[k].keys() ])
        gt_files = sorted_dirs[k] # ground truth files
        
        annot_limit, gt_limit = len(annot_files) - window_size, len(gt_files) - window_size
        i, j = 0, 0
        while i < annot_limit and j < gt_limit:
            annot_page, annot_panel = annot_files[i][:-4].split("_")
            annot_page, annot_panel = int(annot_page), int(annot_panel)
            
            gt_page, gt_panel = gt_files[j][:-4].split("_")
            gt_page, gt_panel = int(gt_page), int(gt_panel)
            
            if gt_page < annot_page or (gt_page == annot_page and gt_panel < annot_panel):
                j += 1
            
            elif gt_page > annot_page or (gt_page == annot_page and gt_panel > annot_panel):
                i += 1
            
            elif gt_page == annot_page and gt_panel == annot_panel:
                if annot_files[i:i+window_size] == gt_files[j:j+window_size]:
                    seq_panels.append([ k + "/" + file for file in annot_files[i:i+window_size ]])
                i += 1
                j += 1
            
            else:
                print("[ERROR] An exception occurred!")
                print("---> GT   :", *gt_files[j:j+window_size])
                print("---> ANNOT:", *annot_files[i:i+window_size])
    
    print("[INFO] Found", len(seq_panels), "sequential panels!")
    return seq_panels


def set_face_area(pw, ph, face_annot, add_margin=False):
    x1, y1, x2, y2 = face_annot
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(pw, x2), min(ph, y2)
    # calculate the centers
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    if add_margin:
        # to get a better center, having hair info also
        cy -= int((y2 - y1) / 6)
    
    radius = max(x2-x1, y2-y1) / 2
    if add_margin:
        radius += radius / 2 # add margin length
    radius = int(radius) # floor to the closest integer
    
    if int(min(pw, ph) / 2) < radius:
        radius = int(min(pw, ph) / 2)
    
    area = [cx-radius, cy-radius, cx+radius, cy+radius]
    
    if area[0] < 0 and area[2]-area[0] <= pw:
        # where x start is on the left side of the image start
        area[2] = area[2]-area[0]
        area[0] = 0
    
    elif area[2] > pw and area[2]-area[0] <= pw:
        # where x end is on the right side of the image end
        area[0] -= area[2] - pw
        area[2] = pw
        
    if area[1] < 0 and area[3]-area[1] <= ph:
        # where y start is on the upper side of the image start
        area[3] = area[3] - area[1]
        area[1] = 0
    
    elif area[3] > ph and area[3]-area[1] <= ph:
        # where y end is on the lower side of the image end
        area[1] -= area[3] - ph
        area[3] = ph
    
    if area[0] < 0 or area[2] > pw or area[1] < 0 or area[3] > ph:
        # There will be no other exceptions since the radius is 
        # limited by the min of the side lengths
        print("[ERROR] For", file, "with coords: [", x1, y1, x2, y2, "] an exception found!")
    
    return area

def select_area_given_face(pw, ph, face_annot, add_margin, w_h_ratio):
    x1, y1, x2, y2 = set_face_area(pw, ph, face_annot, add_margin=add_margin)
    w, h = -1, -1
    if pw / ph >= w_h_ratio:
        w, h = int(ph * w_h_ratio), int(ph)
    elif pw / ph < w_h_ratio:
        w, h = int(pw), int(pw / w_h_ratio)
        
    area = [pw/2 - w/2, ph/2 - h/2, pw/2 + w/2, ph/2 + h/2]
    
    if x2 - x1 > w: 
        # if width of the face is bigger than the selected panel width
        area[0] = (x2+x1)/2 - w/2
        area[2] = (x2+x1)/2 + w/2
    
    elif y2 - y1 > h:
         # if height of the face is bigger than the selected panel height
        area[1] = (y2+y1)/2 - h/2
        area[3] = (y2+y1)/2 + h/2
    
    elif x1 < area[0]:
        # if start of the face is on the left side of the panel width start
        shift_amount = np.random.randint(area[0] - x1, min(area[0], area[2] - x2) + 1)
        area[0] -= shift_amount
        area[2] -= shift_amount
    
    elif x2 > area[2]:
        # if end of the face is on the right side of the panel width end
        shift_amount = np.random.randint(x2 - area[2], min(pw - area[2], x1 - area[0]) + 1)
        area[0] += shift_amount
        area[2] += shift_amount
    
    elif y1 < area[1]:
        # if start of the face is on the upper side of the panel height start
        shift_amount = np.random.randint(area[1] - y1, min(area[1], area[3] - y2) + 1)
        area[1] -= shift_amount
        area[3] -= shift_amount
    
    elif y2 > area[3]:
        # if end of the face is on the right side of the panel width end
        shift_amount = np.random.randint(y2 - area[3], min(ph - area[3], y1 - area[1]) + 1)
        area[1] += shift_amount
        area[3] += shift_amount
        
    return [list(map(int, area)), [x1, y1, x2, y2]]

def select_random_face_and_area(pw, ph, face_annots, add_margin, w_h_ratio):
    idx = np.random.randint(0, len(face_annots))
    return select_area_given_face(pw, ph, face_annots[idx], add_margin, w_h_ratio)


def crop_panels_random(data_dir, annots, seq_panels, add_margin, w_h_ratio):
    """ 
    Crops the panels so that each panel size is maximized and 
    eacn panel has a full face box.
    """
    saved_sizes = {}
    full_coord_data = []
    
    for panels in tqdm(seq_panels):
        for panel in panels:
            # gets width and height information of that paticular panel
            if panel not in saved_sizes:
                w, h = Image.open(os.path.join(data_dir, panel)).size
                saved_sizes[panel] = [w, h]
        
        last_panel = panels[-1]
        annot_folder, annot_details = last_panel.split("/")
        for face_annot in annots[annot_folder][annot_details]:
            
            extracted_coords = [panels, []]
            
            last_panel_data = select_area_given_face(
                *saved_sizes[last_panel], face_annot, add_margin, w_h_ratio)
        
            # processing previous panels separately
            for panel in panels[:-1]:
                annot_folder, annot_details = panel.split("/")
                panel_data = select_random_face_and_area(
                    *saved_sizes[panel], annots[annot_folder][annot_details], add_margin, w_h_ratio)
                
                extracted_coords[1].append(panel_data)
            extracted_coords[1].append(last_panel_data)
            full_coord_data.append(extracted_coords)
    
    return full_coord_data
 
    
def get_panels_and_faces(data_dir, annots, seq_panels, add_margin=False, w_h_ratio=1, method="random"):
    print("[INFO] Extracting Face Coords and Crop Coords for Sequential Panels...")
    if method == "random":
        return crop_panels_random(data_dir, annots, seq_panels, add_margin, w_h_ratio)
    else:
        raise NotImplementedError

def save_data(extracted_data, save_file, shuffle=True):
    extracted = {}
    random.shuffle(extracted_data)   
    for idx, d in enumerate(extracted_data):
        extracted[idx] = d
    
    with open(save_file, 'w') as f:
        json.dump(extracted, f)

        
if __name__ == '__main__':
    # Parameters to set
    golden_dir = "/datasets/COMICS/raw_panel_images/"
    annot_dir = "/userfiles/comics_grp/golden_age/face_annots/"
    save_file = "./panel_face_areas_margin.json"
    extraction_method = "random"
    conf_thold = 0.9
    face_thold = 32
    window_size = 3
    w_h_ratio = 1
    shuffle = True
    add_face_margin = True
    # Extraction Process
    annots = read_golden_annots(annot_dir, conf_thold, face_thold)
    gt_paths = return_folder_structure(golden_dir)
    seq_panels = get_sequential_panels(annots, gt_paths, window_size=window_size)
    extracted_data = get_panels_and_faces(golden_dir, 
                                          annots, 
                                          seq_panels,
                                          add_margin=add_face_margin,
                                          w_h_ratio=w_h_ratio, 
                                          method=extraction_method)
    
    save_data(extracted_data, save_file, shuffle=shuffle)
    
#     for d in extracted_data:
#         print("=========================================================")
#         for i in range(len(d[0])):
#             print(d[0][i], "--->", d[1][i])
#         print("=========================================================\n")
    
    
            
    
     
    
        
    
    
    
        

