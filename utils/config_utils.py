import yaml
from enum import Enum
from collections import namedtuple
from configs.base_config import *


class Config(Enum):
    FACE_RECOGNITION = 1
    BiGAN = 2


def read_config(config: Config):
    if config == Config.FACE_RECOGNITION:
        path = base_dir + 'configs/face_recognition_config.yaml'
    elif config == Config.BiGAN:
        path = base_dir + 'configs/bigan_config.yaml'
    else:
        raise NotImplementedError
    with open(path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    configs = namedtuple("Config", configs.keys())(*configs.values())
    return configs


if __name__ == '__main__':
    res = read_config(Config.BiGAN)
    print(res)
