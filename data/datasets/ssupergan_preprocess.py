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


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.augment import read_image

def sorted_nicely(l): 
    # Alphanumerical sort
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def return_folder_structure(dataset_path =  "/datasets/COMICS"):
    """
    Input : Dataset_path 
    Output : Dictionary that stores folders as key and images as value
    Reason : To have datastructure to check whether given file_name in the folder or not
    """
    print("Creating Folder Structure")

    cur_path = os.path.join(dataset_path, "raw_panel_images")    
    folder_structure = {}
    for folder in tqdm(sorted_nicely(os.listdir(cur_path))):
        folder_path = os.path.join(cur_path, folder)
        folder_structure[folder] = sorted_nicely(os.listdir(folder_path))
        
    return folder_structure
    
def count_heights_widths(dataset_path):
    """
    Reads all the Images and Store Their Width and Height
    This info will be used for padding and cropping operations,
    
    Usage: #height_width_counts = count_heights_widths(dataset_path)
    """ 

    #calculate height and width
    print("Counting and storing width and height information of panels")
    all_panels_info = {}
    for serie in tqdm(os.listdir(os.path.join(dataset_path, "raw_panel_images"))):
        serie_path = os.path.join(os.path.join(dataset_path, "raw_panel_images",serie))
        serie_data = []
        for panel in os.listdir(serie_path):
            panel_path = os.path.join(serie_path, panel)
            cur_image = Image.open(panel_path)
            cur_panel_info = {"widht":cur_image.width, "height":cur_image.height, "serie":serie, "panel" :panel}
            serie_data.append(cur_panel_info)
        all_panels_info[serie] = serie_data
        
    return all_panels_info



def create_annotation_file(valid_sequences):
    
    """
    Input : List of List<seq1, seq2, seq3> 
    Output : Json File to be used as Input to DataLoader
    
    """
    dataset_path = "/datasets/COMICS"
    FOLDER_PATH = os.path.join(dataset_path,"raw_panel_images")
    data = []
    index = 0

    print("Annotation File is being created return DataFrame")
    for (folder, sequences) in tqdm(valid_sequences.items()):
        # Folder "0" , "1" etc.
        for seq in sequences:
            # Sequence --> ['50_3.jpg', '50_4.jpg', '50_5.jpg']
            #List of Annotations 
            #face_annotations["folder"][seq]
            

            data_point = {"index" :index , "folder" : folder, "sequence":seq, "path": os.path.join(FOLDER_PATH, folder) }
            index +=1
            data.append(data_point)
            
    return pd.DataFrame(data)

            

    
 #TODO Find the correct statistics
def count_statistics_height_width(height_widths):
    """ Counts widths Heights of the Panel """ 

    print(" Height and Width Statistics of Panels are being created.")

    #width_stats, height_stats = count_statistics_height_width(height_width_counts)
    width_stats = [ round(element["widht"]/5)*5 for k,v in height_widths.items()  for element in v]
    height_stats = [ round(element["height"]/5)*5 for k,v in height_widths.items()  for element in v]
    
    print(f"Mean of Height : {statistics.mean(height_stats)},  Median of Height {statistics.median(height_stats)}, Mode of Height : {statistics.mode(height_stats)} Num Samples {len(height_stats)}")
    print(f"Mean of Widths  : {statistics.mean(width_stats)},  Median of Height {statistics.median(width_stats)}, Mode of Height : {statistics.mode(width_stats)} Num Samples {len(height_stats)} ")
    
    
    return width_stats, height_stats


def read_images(file_names, folder_path):
    """
    Takes file names and their folder path as input
    Return List of Images
    """
    paths = [os.path.join(folder_path,file) for file in file_names ]
    
    return [read_image(path, augment=False, resize_len=[-1, -1]) for path in paths]



def display_images(images,
    columns=5, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):
    
    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size);
            
            
def find_sequential_panels(panels, window_size):
    """
    Input : Alphanumerical Sorted Panels Names
    Returns : List Of 
    """

    num_panels = len(panels)
    panel_list = []
    for i in range(num_panels-window_size):
        panels_in_window = []
        for k in range(window_size):
            panels_in_window.append(panels[i+k])
        panel_list.append(panels_in_window)

    return panel_list


            
def find_possible_sequnces_all(dataset_path, window_size=3):


    raw_panel_path = os.path.join(dataset_path, "raw_panel_images")
    series = sorted_nicely(os.listdir(raw_panel_path))
    
    data= {}
    for serie in series[:3]:
        serie_information = {}
        serie_path  = os.path.join(raw_panel_path, serie)
        panels =  sorted_nicely(os.listdir(serie_path))
        serie_information["serie_path"] = serie_path
        serie_information["sequential_panels"] = find_sequential_panels(panels, window_size)
        data[serie] = serie_information
    return data


def show_sequence(data, serie_id=0, index_sequence=10):
    serie = data[str(serie_id)]
    images = read_images(serie["sequential_panels"][index_sequence], serie["serie_path"] )
    display_images(images)
    

def calculate_annotation_area(annotation):
    x_min, y_min, x_max, y_max = annotation
    
    height = (y_max - y_min)
    width = (x_max - x_min)
    area = width*height
    return width, height, area



def read_face_detection_annotations(golden_annotations_path, face_size = (32,32)):
    
    """
    Reads Golden Comic Face Annotations and filters Faces with size > (32,32)
    
    Example Usage : 
    golden_annotations_path = "/kuacc/users/ckoksal20/COMP547Project/golden_annot"
    face_annotations,num_small_faces = read_face_detection_annotations(golden_annotations_path)
    
    """

    print("Face Annotations are being readed.")
    ideal_face_size = face_size[0]* face_size[1]
    num_small_faces  = 0
    
    face_annotations = {}
    face_annotation_series_list = os.listdir(golden_annotations_path)

    
    annotations_all = {}
    
    for series in tqdm(sorted_nicely(face_annotation_series_list)): # Series -> 1159.txt
        file_path = os.path.join(golden_annotations_path,series)
        # Reads Annotations
    
        annotations = open(file_path).readlines()
        for annot in annotations:
            
            serie_panel, xmin, ymin, xmax, ymax, confidence = annot.split()
            
            xmin = max(int(xmin),0)
            ymin = max(int(ymin),0)
            xmax = max(int(xmax),0)
            ymax = max(int(ymax),0)
            
            #TODO MAX HEIGHT AND WIDTH     
                
            serie_id, panel = serie_panel.split("/")
            
            #Create_annotation object
            #annotation = Annotation(xmin,ymin,xmax,ymax, confidence, serie_id, panel)
            
            _, _, ann_area = calculate_annotation_area([xmin, ymin, xmax, ymax])
            
            #

            if (ann_area > ideal_face_size)  :#and (float(confidence) > 0.90) :

            
            
                if serie_id not in face_annotations.keys():
                    face_annotations[serie_id] = {}
                    face_annotations[serie_id][panel] = []
                    face_annotations[serie_id][panel].append(torch.tensor([xmin,ymin,xmax,ymax]))
                #Serie_id in face_annotations
                else:
                    if panel not in face_annotations[serie_id].keys():
                        face_annotations[serie_id][panel] = []
                        face_annotations[serie_id][panel].append(torch.tensor([xmin,ymin,xmax,ymax]))
                    else:
                        face_annotations[serie_id][panel].append(torch.tensor([xmin,ymin,xmax,ymax]))
            else:
                #print("Ann area : ",ann_area, "ideal face size : ",ideal_face_size)
                num_small_faces +=1

    return face_annotations,num_small_faces


def write_to_pickle(filename, data):
    # Serialized Writing
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
        
def read_pickle(filename):
    "Reads pickle file and Returns object"
    with open(filename,"rb") as f:
        obj = pickle.load(f)
    return obj


def return_all_possible_panel_seq_face_detection(face_annotations,window_size=3):
    """
    For the Face Annotation Dataset
    Returns all possible combinations of panels with given windows_size
    Use the Data returned by face detector 
    Returns : Dict[serie_id] = List[ Possible Sequence Combinations]
    
    Ex:
      {'1159': [['0_0.jpg', '1_0.jpg', '3_1.jpg'],
      ['1_0.jpg', '3_1.jpg', '3_3.jpg'],
      ['3_1.jpg', '3_3.jpg', '4_0.jpg'],
      ['3_3.jpg', '4_0.jpg', '4_1.jpg'],
      ['4_0.jpg', '4_1.jpg', '4_2.jpg'],
      ['4_1.jpg', '4_2.jpg', '5_0.jpg']...]}
    """

    
    print("Creating all possible panels annotations according to face annotations that came from Face Detector")

    panel_list_face_detection = {}
    for key,folder in tqdm(face_annotations.items()): #dict_keys(['1159', '1955', '2653', '406', '2836', '1289', '1141', 
        #Folder is also dictionary
        current_folder = []
        #print(folder.keys())
        # Jpgs in the folder
        sorted_panels_names = sorted_nicely(folder) ## Jpgs are in alphanumerical order
        #List of annotations
        num_panels = len(sorted_panels_names)
        window_size=3
        #Sliding window with window size 3
        for i in range(num_panels-window_size):
            panels_in_window = []
            for k in range(window_size):
                cur_panel = sorted_panels_names[i+k] #0_0.jpg
                panels_in_window.append(cur_panel)

            current_folder.append(panels_in_window)
        panel_list_face_detection[key] = current_folder
        
    return panel_list_face_detection


def  valid_sequence_creator_from_face_annotations(all_possible_panel_sequence_face_detection, folder_structure):
    """ 
    Input : 
    all_possible_panel_sequence_face_detection  :: Possible Panel Combinations that Face Detector find a panel
    folder_structure : Dictionary that stores folder name as keys and Value --> List of All Jpgs.

    Format --> Key Folder_Id  Value : All possible Sequences
    Goal :  Function Checks whether the sequence is valid or not
    
    Exampe Usage:
    valid_sequences = valid_sequence_creator_from_face_annotations(all_possible_panel_sequence_face_detection, folder_structure)
    
    """

    print("Valid Sequences that are being extracted from all_possible_panel_sequence annotations")
    
    valid_sequences = {}
    
    for i,panel in tqdm(all_possible_panel_sequence_face_detection.items()):
        #Prev_Page_id : 26 prev_panel_id : 4, sequence : ['26_4.jpg', '26_5.jpg', '27_0.jpg']
        valid_sequences_try = []
        for sequence in panel:
            prev_page_id, prev_panel_id =  sequence[0].split(".jpg")[0].split("_")
            #print(f"Prev_Page_id : {prev_page_id} prev_panel_id : {prev_panel_id}, sequence : {sequence}")
            non_valid = False
            for panel in sequence[1:]:
                cur_page_id, cur_panel_id = panel.split(".jpg")[0].split("_")

                #print(f"Cur_page id : {cur_page_id}    cur_panel_id {cur_panel_id} ")
                # Next found face is in the right top page
                if (cur_page_id != prev_page_id) and (cur_page_id == str(int(prev_page_id)+1)):
                    #print("Yey")
                    # Check whether there is gap or not. 
                    # there is no gap
                    expected_panel_op1 = prev_page_id + "_" + str(int(prev_panel_id)+1) +".jpg"
                    expected_panel_op2 = str(int(prev_page_id) + 1) + "_" + "0.jpg"

                    #print(f" ")
                    if (expected_panel_op1 not in folder_structure[i]) and  (expected_panel_op2 == panel):
                        #print(f"Expected_panel_op1 : {expected_panel_op1}, expected_panel_op2 : {expected_panel_op2}  not in folder, there is gap but it is okay ")

                        # For ['38_6.jpg', '39_2.jpg', '39_3.jpg'] case,  is still problematic  ---> Models checks whet
                        prev_page_id = cur_page_id
                        prev_panel_id = cur_panel_id
                    else:
                        #print(f"Expected_panel_op1 : {expected_panel_op1}, expected_panel_op2 : {expected_panel_op2}")
                        #print("Not a valid structure , Before going to next panel there is gap")
                        # Not a valid structure
                        non_valid = True
                        break
                elif (cur_page_id == prev_page_id) and (cur_panel_id ==  str(int(prev_panel_id) + 1)):
                     # everything is fine
                    prev_page_id = cur_page_id
                    prev_panel_id = cur_panel_id

                elif (cur_page_id == prev_page_id) and (cur_panel_id !=  str(int(prev_panel_id) + 1)):

                    expected_panel = prev_page_id + "_" + str(int(prev_panel_id)+1)+".jpg"
                    if (expected_panel not in folder_structure[i]):
                        #print(f"Expected panel not in folder , valid case,  Expected Panel {expected_panel}")
                        prev_page_id = cur_page_id
                        prev_panel_id = cur_panel_id
                    else:
                        #print("Not a valid structure expected panel in the folder but detector could not find face on it ")
                        # Not a valid structure
                        non_valid = True
                        break

                else:
                    #print("ELSE not valid ", "")
                    #print(f"Cur_page id : {cur_page_id}    cur_panel_id {cur_panel_id} ")
                    #print(f"Prev_Page_id : {prev_page_id} prev_panel_id : {prev_panel_id}, sequence : {sequence}")
                    #print("\n")
                    non_valid = True
                    break


            if non_valid != True:
                valid_sequences_try.append(sequence)
            #print("\n")
        #print(i)
        valid_sequences[i] = valid_sequences_try
    return valid_sequences





def crop_face_v2(original_image, face_annotation):
    
    # To have bigger crops 
    H,W, C = original_image.shape
    
    #print("crop_face_v2 \n ", " Orig Image Shape : ",original_image.shape, "Face Annotation Shape : ",face_annotation )
    
    
    ann2 = copy.deepcopy(face_annotation)
    
    xmin,ymin,xmax,ymax  = ann2
    face_width = xmax-xmin
    face_height = ymax-ymin
    
    maximum = face_width
    minimum = face_height
    
    if max(face_height, face_width ) == face_height: 
        maximum = face_height
        minimum = face_width
    else:
        maximum = face_width
        minimum = face_height
        
    crop_pixel = int((maximum - minimum)/2)
    
    
   # print("crop_face_v2 \n ", " Orig Image Shape : ",original_image.shape, "Face Annotation Shape : ",face_annotation, 
          #"face width ", face_width, "face height ",face_height,  "max ",maximum, "min ",minimum, "crop pixel : ",crop_pixel )
    # Long Kenar is Width
    if maximum == face_width:
        #print(" crop_face_v2 Maximum face_width  ")
        
        if ((ymin-crop_pixel) >=0)  and  ((ymax + crop_pixel)<H) :
            
            # Everthing is fine
            
            #print(" crop_face_v2 Maximum face_width  Everythin is fine c ")
            
            face = copy.deepcopy(original_image)[ymin-crop_pixel:ymax +crop_pixel, xmin:xmax,:] 
            annots_old = copy.deepcopy(face_annotation)
            annots = torch.FloatTensor([xmin , ymin-crop_pixel, xmax , ymax + crop_pixel ])
            
        else: 
            
            if ( (ymin-crop_pixel) <0 and ((ymax + crop_pixel)<H)):
                #print("CROP FACE V2  ymin -crop_pizel < 0 ")
                
                bottom_pad = np.abs(ymin-crop_pixel)
                face = copy.deepcopy(original_image)[0:ymax +bottom_pad, xmin:xmax,:] 
                annots_old = copy.deepcopy(face_annotation)
                annots = torch.FloatTensor([xmin , 0, xmax , ymax + bottom_pad ])
                
                
            elif ((ymax + crop_pixel)>H and ((ymin-crop_pixel) >=0)) :
                #print("CROP FACE V2 ymax+crop_pixel > H ")
                # Goes beyond the Height
                go_to_up = crop_pixel + ymax-H
                    
                face = copy.deepcopy(original_image)[ymin - go_to_up :H, xmin:xmax,:] 
                annots_old = copy.deepcopy(face_annotation)
                annots = torch.FloatTensor([xmin , ymin - go_to_up , xmax , H])
                
            else:
                #print("DO NOTHING YAA")
                face = copy.deepcopy(original_image)[ymin :ymax, xmin:xmax,:] 
                annots_old = copy.deepcopy(face_annotation)
                annots = torch.FloatTensor([xmin , ymin  , xmax , ymax])
                
                
                #raise Exception("crop_face_v2 Maximum Face Width  ELSE Third case")
                #raise Exception("crop_face_v2 maximum width Exception Else")
    # Long Kenar is Height
    elif maximum == face_height :
        #print("maximum_width crop_face_v2 ")
        
        # Stays Inside
        if ((xmin-crop_pixel) >=0)  and  ((xmax + crop_pixel)<W) :
            #print("maximum_width crop_face_v2  stays if ")
            #Maximum is panel width 
            face = copy.deepcopy(original_image)[ymin:ymax  , xmin-crop_pixel:xmax+crop_pixel,:] 
            
            annots_old = copy.deepcopy(face_annotation)
            annots = torch.FloatTensor([xmin-crop_pixel  , ymin, xmax +crop_pixel , ymax ])
            
        else:
            
            # Sola Taşma
            if (((xmin-crop_pixel) <0)  and (xmax + crop_pixel)<W ):
                #print("maximum_width crop_face_v2  first if ")
                #print("Info", " Orig Image Shape : ",original_image.shape, "Face Annotation Shape : ",face_annotation, 
              #"face width ", face_width, "face height ",face_height,  "max ",maximum, "min ",minimum, "crop pixel : ",crop_pixel )
                
                
                move_to_right = crop_pixel- np.abs(xmin-crop_pixel)
                face = copy.deepcopy(original_image)[ymin:ymax  , 0:xmax+crop_pixel+move_to_right ,:] 
            
                annots_old = copy.deepcopy(face_annotation)
                annots = torch.FloatTensor([0  , ymin, xmax+crop_pixel+move_to_right , ymax ])
            
            # Sağa Taşma,  Sol Tek Crop_pixel Serbest
            elif( ((xmax + crop_pixel)>W) and ((xmin-crop_pixel) >=0)) :
                
                
                #print("maximum_width crop_face_v2  second if ")
                
                
                space_at_right = W-xmax
                
                needed_space_to_left = 2*crop_pixel - space_at_right
                
                
                if xmin-needed_space_to_left > 0:
                    left_border = xmin - needed_space_to_left
                    right_border = W
                    face = copy.deepcopy(original_image)[ymin:ymax , left_border: W ,:] 
                    
            
                    #annots_old = copy.deepcopy(face_annotation)
                    annots = torch.FloatTensor([(xmin-needed_space_to_left),ymin, W , ymax ])
                    
                else:
                    
                    disari_tasan = abs(xmin-needed_space_to_left)
                    

                    left_border = 0
                    face = copy.deepcopy(original_image)[ymin:ymax , left_border : W ,:] 
                    
                    annots = torch.FloatTensor([0,ymin, W , ymax ])
                
            else:
                #TODOODODODOODODODO
                #raise Exception("crop_face_v2 maximum width Exception Else")
                face = copy.deepcopy(original_image)[ymin:ymax  , xmin:xmax,:] 
            
                annots_old = copy.deepcopy(face_annotation)
                annots = torch.FloatTensor([xmin  , ymin, xmax , ymax ])
        
    #print("Return Annots")
    return face,annots


# Last Image 
def crop_last_image(image, annotation):
    
    """ Crops the Last Panel in the Sequence
        Makes it Square
        Adjust The ANnotation
    
    """
    
    #Find Center
    H,W, C = image.shape
    
    new_annotation = copy.deepcopy(annotation)
    
    xmin,ymin,xmax,ymax = new_annotation
    #xmin,ymin,xmax,ymax = copy.deepcopy(xmin_), copy.deepcopy(ymin_), copy.deepcopy(xmax_), copy.deepcopy(ymax_)
    
    #print("Image shape ",image.shape, "org Ann",annotation)
    maximum = W
    minimum = H
    if max(H,W) == H: 
        maximum = H
        minimum = W
    else:
        maximum = W
        minimum = H
        
    
    crop_pixel = int((maximum - minimum)/2)
    
    #print("CROP LAST IMAGE FUNCTION  INPUT IMAGE SHAPE : ",image.shape, "ANNOTATION", annotation, "MAXIMUM ",maximum, "MINIMUM ",minimum, "CROP PIXEL ",crop_pixel)
    
    # WIDTH is longer -  
    if maximum == W: 
        #print("CROP LAST IMAGE MAX W")
        # Inside
        if (xmin > crop_pixel) and (xmax < W-crop_pixel):
            # Crop
            image = image[:,crop_pixel:W-crop_pixel,: ]
            #Annotation 
            xmin -= crop_pixel
            xmax -= crop_pixel
        else:
            #print("CROP LAST IMAGE MAX W ELSE  --> NOT PROPER CROP")
            # Problematic Case
            # If xmin bigger than crop_pixel Right Side should be cropped
            # Right Crop Does not change annotation of Xmin Xmax
            if (xmin <= crop_pixel and  xmax < W-crop_pixel) :
                
                #print("CROP LAST IMAGE MAX W ELSE  --> NOT PROPER CROP IF ")
                if  (W-2*crop_pixel  >xmax):
                    
                    image = image[:,:W-2*crop_pixel,: ]
                else:
                    
                    iceri_kacan = crop_pixel - xmin
                    
                    #pad = int((W-xmax)/4)
                    border = W-crop_pixel-iceri_kacan
                    if border < xmax:
                        border = xmax
                    image = image[:,xmin:border,: ]
                    
                    xmax = xmax-xmin
                    xmin = 0
                    
                    
            # Left side should be cropped 
            elif  (xmin > crop_pixel) and (xmax >= W-crop_pixel):
                #print("CROP LAST IMAGE MAX W ELSE  --> NOT PROPER CROP ELIF ")
                
                if (2*crop_pixel < xmin) : 
                    image = image[:,2*crop_pixel:,: ]
                    xmin -= 2*crop_pixel
                    xmax -= 2*crop_pixel
                else:
                    #
                    sagdan_gelen = W-xmax
                    solbaslangic =  2*crop_pixel - sagdan_gelen
                    
                    if solbaslangic > xmin:
                        solbaslangic =xmin
                        xmin = 0
                        #pad=  int(xmin/4)
                        image = image[:, solbaslangic: xmax,: ]
                        xmax = xmax-solbaslangic
                    
                    else:
                        image = image[:, solbaslangic: xmax,: ]
                        
                        xmin = xmin-solbaslangic
                        xmax = xmax-solbaslangic
                        
                    #xmax = xmax-xmin+pad
                    #xmin = pad
                    
                
            elif (xmin <= crop_pixel) and (xmax >= W-crop_pixel):
                
                #print("XMİN ",xmin, "Xmax",xmax, "crop pixel ",crop_pixel, "Image Shape" ,image.shape, "Annotation ",annotation)
                
                
                image = image[:, xmin:xmax, : ]
                xmin = 0
                xmax = xmax-xmin
                
                
                #raise Exception("ELSE NOT IMPLEMENTED BOTH CASE")
            else:
                raise Exception("Still hata")
                
    elif maximum == H:
        #print(" CROP LAST IMAGE Maximum H ")
        
        if (ymin >= crop_pixel) and (ymax <= (H-crop_pixel) ):
            #print("H Correct Case ")
            # Inside 
            image = image[crop_pixel:H-crop_pixel,:,:]
            ymin -= crop_pixel
            ymax -= crop_pixel
        else:
            #print("H Problematic Case ")
            # Crop should be from bottom
            
            #Tepeden crop pixel kesemiyorsun ama asagıdan kesiyorsun
            if (ymin < crop_pixel)  and  (ymax <= H-crop_pixel):
                
                
                if ((H-2*crop_pixel)> ymax):
                    # Tamamını asagıdan kesebilirsin
                    image = image[:H-2*crop_pixel,:,:]
                else:
                    #pad = int((H-max)/4)
                    
                    
                    #Öbür taraftan artı olarak kesmek gereken kare yapmak icin
                    pad_top = np.abs(ymin - crop_pixel)
                    
                    image = image[ymin:ymax + pad_top,:,:]
                    
                    ymin = 0
                    ymax = ymax -(crop_pixel-pad_top)
                
            elif ((ymax > (H-crop_pixel)) and (ymin >= crop_pixel)):
                
                
                #print("TTTTTTT ")
                if (ymin >= (2*crop_pixel) ):
                    
                    image  = image[2*crop_pixel:,:,:]
                    ymin -= 2*crop_pixel
                    ymax -= 2*crop_pixel
                else:
                    
                    # Topdan ymine kadar gel
                    
                    #asagidan = (2*crop_pixel)-ymin 
                   #print(" NO PLACE FOR 2X CROP FROM TOP")
                    
                    bottom_border = H - (2*crop_pixel-ymin)
                    
                    
                    if bottom_border > ymax:
                        
                        # No problem
                        image  = image[ymin: bottom_border,:,:]

                        ymax  = ymax-ymin
                        ymin  = 0
                        
                    else: 
                        bottom_border = ymax
                        
                        image  = image[ymin: bottom_border,:,:]

                        ymax  = ymax-ymin
                        ymin  = 0

            elif (ymin < crop_pixel) and (ymax > (H-crop_pixel) ):
                # THE FACE ANNOTATION IS BIGGER THAN CROPPED IMAGE
                
                image  = image[ymin: ymax , xmin :xmax,:]
                
                ymin = 0
                ymax = ymax-ymin
                
                #print("Xmin : ",xmin, "YMIN : ",ymin, "XMAX ",xmax, "YMAX : ",ymax, "CROP PIXEL ",crop_pixel)
                #raise Exception("crop_last_image maximum H  else Exception")
                    

    return image, torch.tensor([xmin, ymin, xmax, ymax])


def resize(image, boxes = [], dims=(300, 300)):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
       
        
    W, H = image.width, image.height
   # print("RESIZE IMAGE : ",image)
   # print("RESIZE INPUT IMAGE : ", image, "BOXES ",boxes)
    
    # Resize image
    new_image = FT.resize(image, dims)
    
    if boxes != [] :
        # Resize bounding boxes
        old_dims = torch.FloatTensor([W, H, W, H])
        new_boxes = boxes / old_dims  # percent coordinates


        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]])
        new_boxes = new_boxes * new_dims
        return new_image, new_boxes
    
    else:
        return new_image
    
    
def display_images_v2(images,
    columns=6, width= 20, height=4, max_images=60, 
    label_wrap_length=50, label_font_size=8):
        
    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    
    bs = images.shape[0]
    
    for i in range(len(images)):
        
        image = images[i,:,:,:]
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image.permute(1,2,0)) 

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 
            
            


    
    
            





        
    

            
    
                    
            

            
            