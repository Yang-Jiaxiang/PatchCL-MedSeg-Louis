import torch
import torchmetrics

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
    # 將輸出進行二值化
    output = (output > threshold).float()
    
    # 計算交集和聯合
    intersection = (output * masks).sum(dim=(2, 3))
    union = (output + masks).sum(dim=(2, 3)) - intersection
    
    # 防止分母為零
    smooth = 1e-6
    miou = ((intersection + smooth) / (union + smooth)).mean().item()
    
    # 計算準確率
    correct = (output == masks).float()
    accuracy = (correct.sum() / correct.numel()).item()
    
    # 計算Dice Coefficient
    dice = ((2 * intersection + smooth) / (output.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smooth)).mean().item()
    
    return miou, accuracy, dice

