import torch
import numpy as np 
import torch.nn.functional as F
from utils.DISLOSS import DiceLoss

class DiceCoefficient:
    def __init__(self):
        self.dice_scores = []

    def add_batch(self, predictions, gts):
        # 初始化 DiceLoss
        dice_loss_fn = DiceLoss()
        
        # 計算 Dice loss
        dice_loss = dice_loss_fn(predictions, gts)
        
        # 計算 Dice coefficient
        dice = 1 - dice_loss
        
        # 添加到列表中
        self.dice_scores.append(dice.item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.dice_scores)).item()

class MeanIOU:
    def __init__(self):
        self.miou_scores = []

    def add_batch(self, predictions, gts):
        # Apply threshold
        predictions = F.sigmoid(predictions)    
        
        # Calculate intersection and union
        intersection = (predictions * gts).sum(dim=(2, 3))
        union = (predictions + gts).sum(dim=(2, 3)) - intersection
        
        # Calculate mIOU
        miou = (intersection + 1e-6) / (union + 1e-6)
        self.miou_scores.append(miou.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.miou_scores)).item()

