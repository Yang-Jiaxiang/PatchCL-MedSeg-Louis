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
        
        # Convert ground truth to one-hot encoding
        gts_one_hot = F.one_hot(gts, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
        gts_one_hot = gts_one_hot[:,1:,:,:]
        
        # Apply sigmoid to predictions if not already applied
        predictions = torch.sigmoid(predictions)
        predictions = predictions[:,1:,:,:]
        
        # Calculate intersection and union
        intersection = (predictions * gts_one_hot).sum(dim=(2, 3))
        union = (predictions + gts_one_hot).sum(dim=(2, 3)) - intersection
        
        # Calculate mIOU
        miou = (intersection + 1e-6) / (union + 1e-6)
        
        self.miou_scores.append(miou.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.miou_scores)).item()

def calculate_metrics(output, masks, threshold=0.5):
    """
    計算 mIoU, Accuracy, 和 Dice Coefficient
    
    參數:
    output: 模型的輸出，形狀為 (batch_size, num_classes, height, width)
    masks: 實際的掩碼，形狀為 (batch_size, num_classes, height, width)
    threshold: 二值化的閾值
    
    返回:
    miou: Mean Intersection over Union
    accuracy: 準確率
    dice: Dice Coefficient
    """
    # Apply threshold
    gts_one_hot = F.one_hot(masks, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()
    gts_one_hot = gts_one_hot[:, 1:, :, :]

    predictions = output[:, 1:, :, :]
    predictions = F.sigmoid(predictions)    

    # Calculate intersection and union
    intersection = (predictions * masks).sum(dim=(2, 3))
    union = (predictions + masks).sum(dim=(2, 3)) - intersection

    # Calculate mIOU
    miou = (intersection + 1e-6) / (union + 1e-6)
       
    
    
    
    # 計算Dice Coefficient
    DICE = DiceLoss()
    dice = DICE(output, masks)
    
    return miou.mean().item(), 1-dice