import torch.nn as nn
import torch
import torch.nn.functional as F

# def discrimination_loss(real_output, fake_output, fake_normal_output=None):
    
#     loss =  real_output.log().mean() + (1 - fake_output).log().mean()
    
#     if fake_normal_output is not None:
#         loss += (1 - fake_normal_output).log().mean()
        
#     return - 0.5 * loss

def discrimination_loss(real_output, fake_output, fake_normal_output=None):
    
    loss =  real_output.mean().log() + (1 - fake_output).mean().log()
    
    if fake_normal_output is not None:
        loss += (1 - fake_normal_output).mean().log()
        
    return - 0.5 * loss