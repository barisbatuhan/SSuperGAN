# Self-Supervised Face Generation using Panel Context Information (SSuperGAN)

This model tries to generate masked faces of the characters given the previous sequential frames. 

## Notes:

This repository is not fully completed!

## Datasets:

- [**Golden Age Comics**](https://digitalcomicmuseum.com/): Includes US comics between 1938 to 1956. The extracted panel images are used, which are retrieved through the study [The Amazing Mysteries of the Gutter](https://arxiv.org/abs/1611.05118).

The whole panel data is processed by a cartoon Face Detector model (which can be found in [here](https://github.com/barisbatuhan/FaceDetector)) by using `mixed_r50` weights and by setting `confidence threshold` to 0.55 and `nms threshold` to 0.2. The following statistics are retrieved from the outputs of the detector model.

- **Total files:** 684885
- **Total faces:** 1063804
- **Faces above 64px:** 309079 / 521089 `(min(width, height) >= 64 / max(width, height) >= 64)`
- **Faces above 128px:** 75111 / 158988 `(min(width, height) >= 128 / max(width, height) >= 128)`
- **Faces above 256px:** 13214 / 27471 `(min(width, height) >= 256 / max(width, height) >= 256)`

## Model Architecture

![gmodel](./images/readme_images/Model.JPG)

## Results

![WIP](./images/readme_images/work_in_progress.JPG)


## Modules

### SSuperGAN Module

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

- To train the SSuperGAN network, you have to specify the following parameters in the `ssupergan_config.yaml` file under the *configs* folder.

```yaml
# Sequential Encoder Configs
lstm_hidden: 256       # hidden size of LSTM module
cnn_embed: 2048        # the output dim retrieved from CNN embedding module
fc_hiddens: []         # sizes of FC layers after LSTM output, if there are any
lstm_dropout: 0        # set to 0 if num_lstm_layers is 1, otherwise set to [0, 0.5]
fc_dropout: 0          # dropout ratio of FC layers if there are any
num_lstm_layers: 1     # lstm num_layers parameter

# GAN Configs
image_dim: 64          # Size of the face image to processs
latent_dim: 256        # Latent z dimension
g_hidden_size: 1024    # Generator hidden size dimension
d_hidden_size: 1024    # Discriminator hidden size dimension
e_hidden_size: 1024    # Encoder hidden size dimension

# GAN Trainig Confings
batch_size: 8
train_epochs: 10

discriminator_lr: 0.0002
discriminator_weight_decay: 0.000025
discriminator_beta_1: 0.5
discriminator_beta_2: 0.999

generator_lr: 0.0002
generator_weight_decay: 0.000025
generator_beta_1: 0.5
generator_beta_2: 0.999

seq_encoder_lr: 0.0002
seq_encoder_weight_decay: 0.000025
seq_encoder_beta_1: 0.5
seq_encoder_beta_2: 0.999
```


### BiGAN Module

- In order to run the module 'bigan_config.yaml' file should be created under configs.
- Example Config:

```yaml
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

```yaml
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

```yaml
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
