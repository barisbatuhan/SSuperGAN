import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import os
import sys

from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_panels import GoldenPanelsDataset
from data.augment import get_PIL_image

from networks.ssupervae import SSuperVAE
from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *
import numpy as np

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    # load the pre-trained Keras model
    global model
    config = read_config(Config.SSUPERVAE)
    #golden_age_config = read_config(Config.GOLDEN_AGE)
    ptu.set_gpu_mode(True)
    
    model = SSuperVAE(config.backbone, 
                latent_dim=config.latent_dim, 
                embed_dim=config.embed_dim,
                use_lstm=config.use_lstm,
                seq_size=config.seq_size,
                decoder_channels=config.decoder_channels,
                gen_img_size=config.image_dim,
                lstm_hidden=config.lstm_hidden,
                lstm_dropout=config.lstm_dropout,
                fc_hidden_dims=config.fc_hidden_dims,
                fc_dropout=config.fc_dropout,
                num_lstm_layers=config.num_lstm_layers,
                masked_first=config.masked_first).cuda() 

    if config.parallel == True:
        net = nn.DataParallel(net)

    load_path = "/scratch/users/gsoykan20/projects/AF-GAN/playground/ssupervae/ckpts/lstm_ssupervae_model-checkpoint-epoch99.pth"
    model.load_state_dict(torch.load(load_path)['model_state_dict'])
    model = model.cuda().eval()

# TODO: manipulate to send data back
# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

# Getting Parameters
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('male'))
    parameters.append(flask.request.args.get('book1'))
    parameters.append(flask.request.args.get('book2'))
    parameters.append(flask.request.args.get('book3'))
    parameters.append(flask.request.args.get('book4'))
    parameters.append(flask.request.args.get('book5'))
    parameters.append(flask.request.args.get('isMarried'))
    parameters.append(flask.request.args.get('isNoble'))
    parameters.append(flask.request.args.get('numDeadRelations'))
    parameters.append(flask.request.args.get('boolDeadRelations'))
    parameters.append(flask.request.args.get('isPopular'))
    parameters.append(flask.request.args.get('popularity'))
    return parameters

def get_model_output(s1, s2, s3):
    s1 = np.expand_dims(s1, axis=0)
    s2 = np.embed_dims(s2, axis=0)
    s3 = np.embed_dims(s3, axis=0)
    