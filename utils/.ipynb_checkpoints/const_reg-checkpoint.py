# +
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.DISLOSS import DiceLoss

# -

def consistency_cost(model,teacher_model,imgs,p_masks):
    dice_loss_fn = DiceLoss()
    
    output1=model(imgs)
    output2=teacher_model(imgs)
    
    dice_loss = dice_loss_fn(predictions, gts)
    loss = 1 - dice_loss
    
    return loss
