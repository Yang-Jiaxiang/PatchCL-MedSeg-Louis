import torch
import torch.nn as nn
import torch.nn.functional as F


    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        # Convert targets to one-hot encoding
        target_one_hot = F.one_hot(targets, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot[:,1:,:,:]
        
        # If your model contains a sigmoid activation layer, comment out the following line
        pred = pred[:,1:,:,:]
        pred = F.sigmoid(pred)
        
        # Flatten label and prediction tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target_one_hot.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice