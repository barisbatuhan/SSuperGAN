import yaml
from enum import Enum
from collections import namedtuple
from configs.base_config import *


class Config(Enum):
    FACE_RECOGNITION = 1
    VAEGAN = 2
    GOLDEN_AGE = 3
    SSUPERGAN = 4
    VAE = 5
    SSUPERVAE = 6
    DCGAN = 7
    INTRO_VAE = 8
    SSUPERDCGAN = 9
    SSUPERGLOBALDCGAN = 10

def read_config(config: Config):
    
    if config == Config.FACE_RECOGNITION:
        path = base_dir + 'configs/face_recognition_config.yaml'
    elif config == Config.SSUPERGAN:
        path = base_dir + 'configs/ssupergan_config.yaml'
    elif config == Config.SSUPERVAE:
        path = base_dir + 'configs/ssupervae_config.yaml'
    elif config == Config.GOLDEN_AGE:
        path = base_dir + 'configs/golden_age_config.yaml'
    elif config == Config.VAE:
        path = base_dir + 'configs/vae_config.yaml'
    elif config == Config.INTRO_VAE:
        path = base_dir + 'configs/intro_vae_config.yaml'
    elif config == Config.DCGAN:
        path = base_dir + 'configs/dcgan_config.yaml'
    elif config == Config.SSUPERDCGAN:
        path = base_dir + 'configs/ssuper_dcgan_config.yaml'
    elif config == Config.SSUPERGLOBALDCGAN:
        path = base_dir + 'configs/ssuper_global_dcgan_config.yaml'
    elif config == Config.VAEGAN:
        path = base_dir + 'configs/vaegan_config.yaml'
    else:
        raise NotImplementedError
    
    with open(path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    configs = namedtuple("Config", configs.keys())(*configs.values())
    return configs

def read_config_from_path(path):
    with open(path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    configs = namedtuple("Config", configs.keys())(*configs.values())
    return configs

if __name__ == '__main__':
    res = read_config(Config.BiGAN)
    print(res)
