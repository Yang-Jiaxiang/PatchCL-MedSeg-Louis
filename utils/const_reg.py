# +
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

criterion = smp.utils.losses.CrossEntropyLoss()


# -

def consistency_cost(model,teacher_model,imgs,p_masks):
    output1=model(imgs)
    output2=teacher_model(imgs)
    loss=criterion(output1, output2.argmax(dim=1))
#     loss=F.cross_entropy(output1,output2)
#     loss = F.mse_loss(output1, output2)
    return loss
