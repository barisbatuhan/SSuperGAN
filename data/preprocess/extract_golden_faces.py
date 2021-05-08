import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

os.path.dirname(sys.executable)
sys.path.append("/scratch/users/baristopal20/SSuperGAN/")


def read_golden_annots(annot_dir, conf_thold=0.85, face_thold=64, max_classes=3958):
    
    annots = {}
    for i in range(max_classes + 1):
        # read the annot file
        file_path = os.path.join(annot_dir, str(i) + ".txt")
        if not os.path.exists(file_path):
            continue
        
        lines = open(file_path, "r").readlines()
        for line in lines:
            if len(line) < 2:
                # when there is only newline character
                continue
            
            # read the line annots
            fname, x1, y1, x2, y2, conf = line[:-1].split(" ")
            fname = fname[:-4] # remove the .txt or .jpg file format
            x1, y1, x2, y2, conf = float(x1), float(y1), float(x2), float(y2), float(conf)
     
            if conf < conf_thold:
                continue  
            elif max(x2-x1, y2-y1) < face_thold:
                continue
            
            if fname not in annots:
                annots[fname] = []
            annots[fname].append([x1, y1, x2, y2])
            
    return annots

def extract_faces(annots, golden_dir, save_dir):
    
    for file in tqdm(annots.keys()):
        series, pages = file.split("/")
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        if not os.path.exists(os.path.join(save_dir, series)):
            os.mkdir(os.path.join(save_dir, series))
        
        img = Image.open(os.path.join(golden_dir, file + ".jpg")).convert('RGB')
        w, h = img.size
        
        ctr = 0
        for coord in annots[file]:
            x1, y1, x2, y2 = coord
            # eliminate overflows in the area
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2) 
            # calculate the centers
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cy -= int((y2 - y1) / 6) # to get a better center, having hair info also
            
            radius = max(x2-x1, y2-y1) / 2
            radius += radius / 2 # add margin length
            radius = int(radius) # floor to the closest integer
            
            if int(min(w, h) / 2) < radius:
                radius = int(min(w, h) / 2)
            
            area = [cx-radius, cy-radius, cx+radius, cy+radius]
            
            if area[0] < 0 and area[2]-area[0] <= w:
                # where x start is on the left side of the image start
                area[2] = area[2]-area[0]
                area[0] = 0
            
            elif area[2] > w and area[2]-area[0] <= w:
                # where x end is on the right side of the image end
                area[0] -= area[2] - w
                area[2] = w
                
            if area[1] < 0 and area[3]-area[1] <= h:
                # where y start is on the upper side of the image start
                area[3] = area[3]-area[1]
                area[1] = 0
            
            elif area[3] > h and area[3]-area[1] <= h:
                # where y end is on the lower side of the image end
                area[1] -= area[3] - h
                area[3] = h
            
            if area[0] < 0 or area[2] > w or area[1] < 0 or area[3] > h:
                # There will be no other exceptions since the radius is 
                # limited by the min of the side lengths
                print("[ERROR] For", file, "with coords: [", x1, y1, x2, y2, "] an exception found!")
            
            cropped_img = img.copy().crop(area)
            cropped_img.save(os.path.join(save_dir, series, pages + "_" + str(ctr) + ".jpg"))
            ctr += 1
            
            
if __name__ == '__main__':
    golden_dir = "/datasets/COMICS/raw_panel_images/"
    annot_dir = "/userfiles/comics_grp/golden_annot/"
    save_dir = "./golden_faces/"
    conf_thold = 0.9
    face_thold = 96
    annots = read_golden_annots(annot_dir, conf_thold=conf_thold, face_thold=face_thold)
    extract_faces(annots, golden_dir, save_dir)
    

