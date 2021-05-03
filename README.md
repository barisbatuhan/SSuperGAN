# Self-Supervised Face Generation using Panel Context Information (SSuperGAN)

Using [Golden Age Comics](https://digitalcomicmuseum.com/) data, this model tries to generate masked faces of the characters given the previous sequential frames. 

## Notes:

This repository is not fully completed!

## Model Architecture

![gmodel](./images/readme_imgs/Model.JPG)

## Results

![WIP](./images/readme_imgs/work_in_progress.JPG)


## Modules

### Face Recognition Module

- In order to run the module 'face_recognition_config.yaml' file should be created under configs.
- Example Config:

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
### Project Based Configuration

One should check and update 'configs/base_config' for global config parameters such base project directory.