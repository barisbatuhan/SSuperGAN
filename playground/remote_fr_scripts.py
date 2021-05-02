import os
import sys
os.path.dirname(sys.executable)
sys.path.append('/scratch/users/gsoykan20/projects/AF-GAN/')
from face_recognition import *
from utils.datetime_utils import get_dt_string


def train_fr():
    initiate_logger()
    ptu.set_gpu_mode(True)
    model = train_siamese(get_dt_string() + "_model")
    compute_mean_acc(model, datasource_mode = DataSourceMode.TEST)

if __name__ == '__main__':
    train_fr()