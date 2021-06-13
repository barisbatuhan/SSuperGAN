import base64

import flask
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from flask import request, jsonify
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import os
import sys

from torchvision.transforms import transforms

from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_panels import GoldenPanelsDataset
from data.augment import get_PIL_image, normalize

from networks.ssupervae import SSuperVAE
from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *
import numpy as np
from PIL import Image

# Source: https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1

# initialize our Flask application and the model
app = flask.Flask(__name__)


def init():
    # load the pre-trained Keras model
    global model
    config = read_config(Config.SSUPERVAE)
    # golden_age_config = read_config(Config.GOLDEN_AGE)
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
        model = nn.DataParallel(model)

    load_path = "/home/gsoykan20/Desktop/AF-GAN/playground/ssupervae/ckpts/lstm_ssupervae_model-checkpoint-epoch99.pth"
    model.load_state_dict(torch.load(load_path)['model_state_dict'])
    model = model.cuda().eval()


# API for prediction
@app.route("/predict", methods=["POST"])
def predict():
    panels = list(map(read_image_from_request, ['panel1', 'panel2', 'panel3']))
    output_img = get_model_output(panels[0], panels[1], panels[2])
    img_base64 = base64.b64encode(output_img.tobytes())
    return jsonify({'status': str(img_base64)})


def read_image_from_request(img_name):
    file = request.files[img_name].read()  ## byte file
    npimg = np.fromstring(file, np.uint8)
    img = Image.fromarray(np.uint8(npimg)).convert('RGB')
    img = transforms.ToTensor()(img).unsqueeze(0)
    normalize(img)
    img = img.resize_(3, 256, 256)
    return img.cpu().numpy()


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


@torch.no_grad()
def get_model_output(s1, s2, s3):
    s1 = np.expand_dims(s1, axis=0)
    s2 = np.expand_dims(s2, axis=0)
    s3 = np.expand_dims(s3, axis=0)
    x = np.concatenate([s1, s2, s3], axis=0)
    x = torch.from_numpy(x).cuda()
    _, _, _, y_recon, _ = model(x=x.view(1, *x.size()))
    return get_PIL_image(y_recon.squeeze())


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading SSuperVAE model and Flask starting server..."
           "please wait until server has fully started"))
    init()
    app.run(threaded=True)
