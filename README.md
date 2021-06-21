# Self-Supervised Face Generation using Panel Context Information (SSuperGAN)

This model tries to generate masked faces of the characters given the previous sequential frames. 

## Notes:

This repository is not fully completed!

## Datasets:

- [**Golden Age Comics**](https://digitalcomicmuseum.com/): Includes US comics between 1938 to 1956. The extracted panel images are used, which are retrieved through the study [The Amazing Mysteries of the Gutter](https://arxiv.org/abs/1611.05118).

The whole panel data is processed by a cartoon Face Detector model (which can be found in [here](https://github.com/barisbatuhan/FaceDetector)) by using `mixed_r50` weights and by setting `confidence threshold` to 0.9 and `nms threshold` to 0.2. The following statistics are retrieved from the data:

- ** Total files:** 1229664
- **Panel Height:** mean=510.0328 / median=475 / mode=445
- **Panel Width:** mean=508.4944 / median=460 / mode=460

## Model Architecture

![gmodel](./images/readme_images/Model.PNG)

## Results

### Visual Results

![Result 1](./images/readme_images/res1.PNG)

![Result 2](./images/readme_images/res2.PNG)

### Metric Results

![WIP](./images/readme_images/Results.PNG)

## Pretrained Models and Links

- Face detection (Siamese) on iCartoonDataface (~%86 test acc) [link](https://drive.google.com/file/d/1ey896AyT-uqQ5YlHSp4880da40-Ju1pS/view?usp=sharing)
- [Google Sheet](https://docs.google.com/spreadsheets/d/1JPdPtDocE8LMN4v246cLKqqJB9qZQNbMOtdg1fHy8AI/edit?usp=sharing) for recording Experiment Results

## Modules

### USING GOLDEN AGE DATA

- In order to run the module 'golden_age_config.yaml' file should be created under configs.
- Example Config:

```yaml
# For directly face generation task
faces_path: /userfiles/comics_grp/golden_age/golden_faces_no_margin/
face_train_test_ratio: 0.9

# For panel face reconstruction task
panel_path: /datasets/COMICS/raw_panel_images/
sequence_path: /userfiles/comics_grp/golden_age/panel_face_areas_new_90.json
annot_path: /userfiles/comics_grp/golden_age/golden_annot_new/
mask_val: 1
mask_all: False
return_mask: True
return_mask_coordinates: True
train_test_ratio: 0.95
train_mode: True
```

### SSuper Global DCGAN Module

- In order to run the module 'ssuper_dcgan_config.yaml' file should be created under configs.
- Example Config:

```yaml
# Encoder Parameters
backbone: "efficientnet-b5"
embed_dim: 1024
latent_dim: 512
use_lstm: True

# Plain Encoder Parameters
seq_size: 3

# LSTM Encoder Parameters
lstm_conv: False
lstm_hidden: 1024
lstm_bidirectional: True

# These do not change depending on Conv-LSTM
lstm_dropout: 0
fc_hidden_dims: []
fc_dropout: 0
num_lstm_layers: 1
masked_first: True

# DCGAN Parameters
img_size: 64
panel_size:
    - 300
    - 300
gen_channels: 64
local_disc_channels: 64
global_disc_channels: 64

# Training Parameters
batch_size: 32
train_epochs: 200
lr: 0.0002
weight_decay: 0.000025
beta_1: 0.5
beta_2: 0.999
g_clip: 100

local_disc_lr: 0.0002
global_disc_lr: 0.005
disc_mom: 0.9

# Parallelization Parameters
parallel: True
```

### VAE-GAN Module

- In order to run the module 'vaegan_config.yaml' file should be created under configs.
- Example Config:

```yaml
latent_dim: 512

# DCGAN Parameters
img_size: 64
gen_channels: 64
enc_channels: 
    - 64
    - 128
    - 256
    - 512
local_disc_channels: 64

# Training Parameters
batch_size: 64
train_epochs: 200
lr: 0.0002
weight_decay: 0.000025
beta_1: 0.5
beta_2: 0.999
g_clip: 100

local_disc_lr: 0.0002

# Parallelization Parameters
parallel: True
```

### Plain SsuperVAE Module

- To train the PlainSSuperVAE network, you have to specify the following parameters in the `ssupervae_config.yaml` file under the *configs* folder. To use the LSTM structure, simply set the flag `use_lstm` to `True`.

```yaml
# Fill with new parameters
```

### VAE Module

- In order to run the module 'vae_config.yaml' file should be created under configs.
- Example Config:

```yaml
# Fill with new parameters
```

### Intro-VAE Module

- In order to run the module 'intro_vae_config.yaml' file should be created under configs.
- Example Config:

```yaml
# Fill with new parameters
```

### DCGAN Module

- In order to run the module 'dcgan_config.yaml' file should be created under configs.
- Example Config:

```yaml
# Fill with new parameters
```

### Project Based Configuration

One should check and update 'configs/base_config' for global config parameters such base project directory.
