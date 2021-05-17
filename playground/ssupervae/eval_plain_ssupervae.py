import os
import sys
import json
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader

from data.datasets.golden_panels import GoldenPanelsDataset
from networks.plain_ssupervae import PlainSSuperVAE

from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *

from functional.metrics.psnr import PSNR
from functional.metrics.fid import FID

metrics = ["PSNR", "FID"]

METRIC = metrics[1]
BATCH_SIZE = 8
model_path = "playground/ssupervae/checkpoints/11-05-2021-12-35-34_model-checkpoint-epoch85.pth"
N_SAMPLES = 1280 # 50000

ptu.set_gpu_mode(True)
config = read_config(Config.PLAIN_SSUPERVAE)
golden_age_config = read_config(Config.GOLDEN_AGE)


net = PlainSSuperVAE(config.backbone, 
                     latent_dim=config.latent_dim, 
                     embed_dim=config.embed_dim,
                     seq_size=config.seq_size,
                     decoder_channels=config.decoder_channels,
                     gen_img_size=config.image_dim).to(ptu.device) 

net.load_state_dict(torch.load(model_path)['model_state_dict'])
net = net.cuda().eval()

if METRIC == "PSNR":
    dataset = GoldenPanelsDataset(golden_age_config.panel_path,
                                  golden_age_config.sequence_path, 
                                  golden_age_config.panel_dim,
                                  config.image_dim, 
                                  augment=False, 
                                  mask_val=1, # mask with white color for 1 and black color for 0
                                  mask_all=False, # masks faces from all panels and returns all faces
                                  return_mask=True,
                                  train_test_ratio=golden_age_config.train_test_ratio,
                                  train_mode=False,
                                  limit_size=-1)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    psnrs, iter_cnt = 0, 0
    for x, y, z in tqdm(data_loader):
        with torch.no_grad():
            _, _, _, y_recon, _ = net(x.cuda())
        psnrs += PSNR.__call__(y_recon.cpu(), y, fit_range=True)
        iter_cnt += 1
    print("-- PSNR:", psnrs.item()/iter_cnt)
    
    
elif METRIC == "FID":
    metric = FID(n_samples=N_SAMPLES, batch_size=BATCH_SIZE)
    fid = metric.__call__(net)
    print("-- FID:", fid, "on", N_SAMPLES, "samples.")

