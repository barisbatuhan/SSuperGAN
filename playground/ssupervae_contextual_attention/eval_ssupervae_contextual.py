import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from functional.metrics.psnr import PSNR
from functional.metrics.fid import FID

from torch.utils.data import DataLoader
from data.datasets.golden_panels import GoldenPanelsDataset
from networks.ssupervae_contextual_attentional import SSuperVAEContextualAttentional
from utils.config_utils import read_config, Config
from utils.image_utils import *

import torchvision.transforms.functional as TF

metrics = ["PSNR", "FID"]

METRIC = metrics[0]
BATCH_SIZE = 128 if METRIC == "FID" else 32
N_SAMPLES = 50000

# model_path = "/userfiles/comics_grp/pretrained_models/plain_ssupervae_epoch85.pth"
# use_lstm = False

# model_path = "/userfiles/comics_grp/pretrained_models/lstm_ssupervae_epoch99.pth"
model_path = "/scratch/users/gsoykan20/projects/AF-GAN/playground/ssupervae/checkpoints/25-05-2021-00-01-41_model-checkpoint-epoch51.pth"
use_lstm = False

# Required for FID, if not given, then calculated from scratch
mus = None
sigmas = None

ptu.set_gpu_mode(True)
config = read_config(Config.VAE_CONTEXT_ATTN)
golden_age_config = read_config(Config.GOLDEN_AGE)
panel_dim = golden_age_config.panel_dim[0]

net = SSuperVAEContextualAttentional(config.backbone,
                                     panel_img_size=panel_dim,
                                     latent_dim=config.latent_dim,
                                     embed_dim=config.embed_dim,
                                     seq_size=config.seq_size,
                                     decoder_channels=config.decoder_channels,
                                     gen_img_size=config.image_dim).to(ptu.device)

net.load_state_dict(torch.load(model_path)['model_state_dict'])
net = net.cuda().eval()

dataset = GoldenPanelsDataset(golden_age_config.panel_path,
                              golden_age_config.sequence_path,
                              golden_age_config.panel_dim,
                              config.image_dim,
                              augment=False,
                              mask_val=golden_age_config.mask_val,
                              mask_all=golden_age_config.mask_all,
                              return_mask=golden_age_config.return_mask,
                              return_mask_coordinates=golden_age_config.return_mask_coordinates,
                              train_test_ratio=golden_age_config.train_test_ratio,
                              train_mode=False,
                              limit_size=-1)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

# TEMP
stds = torch.Tensor([0.229, 0.224, 0.225])
means = torch.Tensor([0.485, 0.456, 0.406])


def convert_to_old_augmentation(img):
    img = (img + 1) / 2
    img = TF.normalize(img, mean=means, std=stds)
    return img


def convert_to_new_augmentation(img):
    img = img * stds.view(1, 3, 1, 1).cuda() + means.view(1, 3, 1, 1).cuda()
    img *= 2
    img -= 1
    return img


# TEMP

if METRIC == "PSNR":

    psnrs, iter_cnt = 0, 0
    for x, y, z, mask_coordinates in tqdm(data_loader):
        x_plain = convert_to_old_augmentation(x)
        with torch.no_grad():
            _, _, _, y_recon, _ = net(x_plain.cuda())
        psnrs += PSNR.__call__(convert_to_new_augmentation(y_recon).cpu(), y, fit_range=True)
        iter_cnt += 1
    print("-- PSNR:", psnrs.item() / iter_cnt)


elif METRIC == "FID":

    metric = FID(n_samples=N_SAMPLES, batch_size=BATCH_SIZE)

    if mus is None or sigmas is None:
        iter_cnt = 0
        for _, y, _, mask_coordinates in tqdm(data_loader):
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
