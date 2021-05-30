import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json

from tqdm import tqdm

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.models import SSuperDCGAN

from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *

from configs.base_config import *

from functional.metrics.psnr import PSNR
from functional.metrics.fid import FID

metrics = ["PSNR", "FID"]

METRIC = metrics[1]
BATCH_SIZE = 256 if METRIC == "FID" else 16
model_path = "playground/ssuper_dcgan/ckpts/ssuper_dcgan-checkpoint-epoch36.pth"

N_SAMPLES = 50000 # 50000

# Required for FID, if not given, then calculated from scratch
mus = None
sigmas = None

config = read_config(Config.SSUPERDCGAN)
golden_age_config = read_config(Config.GOLDEN_AGE)

net = SSuperDCGAN(backbone=config.backbone,
                  embed_dim=config.embed_dim,
                  latent_dim=config.latent_dim,
                  img_size=config.img_size,
                  use_lstm=config.use_lstm,
                  gen_channels=config.gen_channels,
                  local_disc_channels=config.local_disc_channels,
                  seq_size=config.seq_size,
                  lstm_bidirectional=config.lstm_bidirectional,
                  lstm_hidden=config.lstm_hidden,
                  lstm_dropout=config.lstm_dropout,
                  fc_hidden_dims=config.fc_hidden_dims,
                  fc_dropout=config.fc_dropout,
                  num_lstm_layers=config.num_lstm_layers,
                  masked_first=config.masked_first)

if config.parallel:
    net = nn.DataParallel(net)

net.load_state_dict(torch.load(model_path)['model_state_dict'])
net = net.cuda().eval()

dataset = GoldenPanelsDataset(golden_age_config.panel_path,
                              golden_age_config.sequence_path, 
                              config.panel_size,
                              config.img_size, 
                              augment=False, 
                              mask_val=1, # mask with white color for 1 and black color for 0
                              mask_all=False, # masks faces from all panels and returns all faces
                              return_mask=True,
                              train_test_ratio=0.01,
                              train_mode=False,
                              limit_size=-1)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

if METRIC == "PSNR":
    
    psnrs, iter_cnt = 0, 0
    for x, y, z in tqdm(data_loader):
        with torch.no_grad():
            mu_z, _ = net(x.cuda(), f="seq_encode")
            mu_z = mu_z.unsqueeze(-1).unsqueeze(-1)
            y_recon = net(mu_z, f="generate")
        psnrs += PSNR.__call__(y_recon.cpu(), y, fit_range=True)
        iter_cnt += 1
    print("-- PSNR:", psnrs.item()/iter_cnt)
    
    
elif METRIC == "FID":
    
    metric = FID(n_samples=N_SAMPLES, batch_size=BATCH_SIZE)
    
    if mus is None or sigmas is None:
        iter_cnt = 0
        for _, y, _ in tqdm(data_loader):
            original_features = metric.extract_features(y).cpu().numpy()
            mu = np.mean(original_features, axis=0)
            sigma = np.cov(original_features, rowvar=False)
            
            if mus is None:
                mus, sigmas = mu, sigma
            else:
                mus += mu
                sigmas += sigma
            
            iter_cnt += 1
            
        mus /= iter_cnt
        sigmas /= iter_cnt
        
    fid = metric.__call__(net, real_mean=mus, real_cov=sigmas)
    print("-- FID:", fid, "on", N_SAMPLES, "samples.")

