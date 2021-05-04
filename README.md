# Self-Supervised Face Generation using Panel Context Information (SSuperGAN)

Using [Golden Age Comics](https://digitalcomicmuseum.com/) data, this model tries to generate masked faces of the characters given the previous sequential frames. 

## Notes:

This repository is not fully completed!

## Model Architecture

![gmodel](./images/readme_images/Model.JPG)

## Results

![WIP](./images/readme_images/work_in_progress.JPG)


## Modules

### SStyleGAN Module

- An example code piece to run a forward pass with SStyleGAN model:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import numpy as np

from networks.ssupergan import SSuperGAN

seq_args = {
    "lstm_hidden": 256,    # Hidden size of LSTM module
    "embed": 256,          # Last size for mean and std outputs
    "cnn_embed": 2048,     # The output dim retrieved from CNN embedding module
    "fc_hiddens": [],      # Sizes of FC layers after LSTM output, if there are any
    "lstm_dropout": 0,     # Set to 0 if num_lstm_layers is 1, otherwise set to [0, 0.5]
    "fc_dropout": 0,       # Dropout ratio of FC layers if there are any
    "num_lstm_layers": 1   # Number of stacked LSTM layers
}

gen_img_dim = 64           # Dimension of the generated image

model = SSuperGAN(seq_args, image_dim=gen_img_dim).cuda()

C, W, H = 3, 360, 360
x = torch.randn(16, 3, C, H, W).cuda()                    # Sample sequential images
y = torch.randn(16, 3, gen_img_dim, gen_img_dim).cuda()   # Sample target images to generate

with torch.no_grad():
    kl, recon, discr = model(x, y)

print("KL Loss:", kl)
print("Reconstruction Loss:", recon)
print("Discriminator Loss:", discr)
```

### BiGAN Module

- In order to run the module 'bigan_config.yaml' file should be created under configs.
- Example Config:
```
face_image_folder_train_path: /home/gsoykan20/Desktop/ffhq_thumbnails/thumbnails128x128/
face_image_folder_test_path: /home/gsoykan20/Desktop/ffhq_thumbnails/thumbnails128x128/
num_training_samples: 10240
test_samples_range:
    - 10240
    - 10640
image_dim: 32
batch_size: 32
train_epochs: 10
discriminator_lr: 0.0002
discriminator_weight_decay: 0.000025
discriminator_beta_1: 0.5
discriminator_beta_2: 0.999
generator_lr: 0.0002
generator_weight_decay: 0.000025
generator_beta_1: 0.5
generator_beta_2: 0.999
```

### USING GOLDEN AGE FACE DATA
- In order to run the module 'golden_age_face_config.yaml' file should be created under configs.
- Example Config:
```
annotations_folder_path: /home/gsoykan20/Desktop/amazing-mysteries-gutter-comics/golden_annot
panel_folder_path: /home/gsoykan20/Desktop/amazing-mysteries-gutter-comics/comics/data/raw_panel_images
min_original_face_dim: 64
num_training_samples: 30000
num_test_samples: 1024
image_dim: 64
```

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
