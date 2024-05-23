import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCELoss(nn.Module):
    def __init__(self):
        super(CustomCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target):
        # 檢查 prediction 的值域範圍
        min_val, max_val = torch.min(prediction).item(), torch.max(prediction).item()

        target = target.float()
        loss = self.bce_loss(prediction, target)
        return loss