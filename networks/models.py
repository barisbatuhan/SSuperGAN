import torch
import torch.nn as nn
from torch import Tensor
import torchvision

# Models
from networks.ssuper_model import SSuperModel

# Helpers
from utils import pytorch_util as ptu

class DCGAN(SSuperModel):
    
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=False, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice=None, **kwargs)
        self.criterion = nn.BCELoss()
    
    def bce_loss(self, output, label):
        return self.criterion(output, label)

class IntroVAE(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=False, enc_choice="vae", gen_choice="vae",
                         local_disc_choice=None, global_disc_choice=None, **kwargs)


class SSuperVAE(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="vae",
                         local_disc_choice=None, global_disc_choice=None, **kwargs)
        
class SSuperDCGAN(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice=None, **kwargs)

class SSuperGlobalDCGAN(SSuperModel):
    def __init__(self, **kwargs):
        super().__init__(use_seq_enc=True, enc_choice=None, gen_choice="dcgan",
                         local_disc_choice="dcgan", global_disc_choice="dcgan", **kwargs)

        