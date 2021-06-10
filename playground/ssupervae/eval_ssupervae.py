import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.ssupervae import SSuperVAE

from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *

from functional.metrics.psnr import PSNR
from functional.metrics.fid import FID

metrics = ["PSNR", "FID"]

METRIC = metrics[1]
BATCH_SIZE = 512 if METRIC == "FID" else 64
N_SAMPLES = 10000

# model_path = "/userfiles/comics_grp/pretrained_models/plain_ssupervae_epoch85.pth"
# use_lstm = False

# model_path = "/userfiles/comics_grp/pretrained_models/lstm_ssupervae_epoch99.pth"
model_path = "/scratch/users/gsoykan20/projects/AF-GAN/playground/ssupervae/ckpts/lstm_ssupervae_model-checkpoint-epoch99.pth"
use_lstm = True

# Required for FID, if not given, then calculated from scratch
mus = None
sigmas = None

ptu.set_gpu_mode(True)
config = read_config(Config.SSUPERVAE)
golden_age_config = read_config(Config.GOLDEN_AGE)

net = SSuperVAE(config.backbone, 
                latent_dim=config.latent_dim, 
                embed_dim=config.embed_dim,
                use_lstm=use_lstm,
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

net.load_state_dict(torch.load(model_path)['model_state_dict'])

if getattr(config, 'parallel', False):
    net = net.module
    
net = net.cuda().eval()

dataset = GoldenPanelsDataset(golden_age_config.panel_path,
                              golden_age_config.sequence_path, 
                              golden_age_config.panel_dim,
                              config.image_dim, 
                              augment=False, 
                              mask_val=1, # mask with white color for 1 and black color for 0
                              mask_all=False, # masks faces from all panels and returns all faces
                              return_mask=True,
                              train_test_ratio= golden_age_config.train_test_ratio,
                              train_mode=False,
                              limit_size=-1)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

if METRIC == "PSNR":
    
    psnrs, iter_cnt = 0, 0
    for x, y, z in tqdm(data_loader):
        with torch.no_grad():
            _, _, _, y_recon, _ = net(x.cuda())
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

