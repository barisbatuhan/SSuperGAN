import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np


def read_image(img_path, augment=True, resize_len=[128, 128]):
    img = Image.open(img_path).convert('RGB')

    if augment:
        img = distort_color(img)
        # img = horizontal_flip(img, box)

    img = resize(img, resize_len)
    img = normalize(transforms.ToTensor()(img))
    return img


def normalize(img, means=0.5, stds=0.5):
    # means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]
    return TF.normalize(img, mean=means, std=stds)


def distort_color(img):
    if np.random.rand() > 0.5:
        br_strength = np.random.randint(4, 21) / 10
        img = TF.adjust_brightness(img, br_strength)  # 0.4 - 2.0 range
    if np.random.rand() > 0.5:
        con_strength = np.random.randint(4, 31) / 10
        img = TF.adjust_contrast(img, con_strength)  # 0.4 - 3.0 range
    if np.random.rand() > 0.5:
        hue_strength = np.random.randint(-5, 5) / 10
        img = TF.adjust_hue(img, hue_strength)  # -0.4 - 0.4 range
    if np.random.rand() > 0.5:
        sat_strength = np.random.randint(-1, 20) / 10
        img = TF.adjust_saturation(img, sat_strength)  # 0.0 - 2.0 range
    return img


def horizontal_flip(img):
    if np.random.rand() > 0.5:
        img = TF.hflip(img)
    return img


def resize(img, resize_len):
    if resize_len[0] != -1:
        img = TF.resize(img, (resize_len[1], resize_len[0]))
    return img


def crop(img, x1, y1, w, h):
    return TF.crop(img, x1, y1, w, h)


def random_crop(img):
    W, H = img.size
    min_len = min(W, H)
    crop_ratios = [0.3, 0.45, 0.6, 0.8, 1.0]

    while True:  # at least center of one face will be in cropped image
        crop_ratio = np.random.choice(crop_ratios)
        side_len = min_len * crop_ratio - 1
        w_start = np.random.randint(0, max(1, W - side_len + 1))
        h_start = np.random.randint(0, max(1, H - side_len + 1))

    img = TF.crop(img, h_start, w_start, side_len, side_len)
    return img


def get_PIL_image(img_tensor, means=0.5, stds=0.5):
    img = img_tensor.to("cpu")

    if means is None or stds is None:
        return transforms.ToPILImage()(img)
    else:
        return transforms.ToPILImage()(img * stds + means)
