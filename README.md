# AF-GAN
AnimeFace-GAN (AF-GAN) : Generating New Faces of the Same Character

# Face Recognition Module

- In order to run the module 'face_recognition_config.yaml' file should be created under configs.
- Exmple Config:

```
face_image_folder_train_path: /datasets/iCartoonFace2020/personai_icartoonface_rectrain/icartoonface_rectrain
face_image_folder_test_path: /datasets/iCartoonFace2020/personai_icartoonface_rectrain/icartoonface_rectrain
num_training_samples: 1024
test_samples_range: 
    - 388000
    - 389678
image_dim: 64
batch_size: 128
train_epochs: 5

Comment: > 
    face_image_folder_path: /home/gsoykan20/Desktop/amazing-mysteries-gutter-comics/comics/data/raw_panel_images
    TODO: Add VALIDATION PATH
    total number of images in icf train set : 389678
    num_training_samples: 20480
    test_samples_range: 
        - 20480
        - 22528
```
