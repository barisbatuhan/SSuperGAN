import torch

class PSNR:
    """
    Peak Signal to Noise Ratio
    retrieved from: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    """

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, fit_range=False):
        
        if fit_range:
            # if input is a tensor between [-1, 1]
            img1 = 255 * (torch.clamp(img1, min=-1, max=1) + 1) / 2
            img2 = 255 * (torch.clamp(img2, min=-1, max=1) + 1) / 2
        
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))