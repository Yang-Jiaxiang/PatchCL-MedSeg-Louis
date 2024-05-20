import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LORALayer(nn.Module):
    def __init__(self, adapted_layer, rank=16):
        super(LORALayer, self).__init__()
        self.adapted_layer = adapted_layer
        self.A = nn.Parameter(torch.randn(adapted_layer.weight.size(1), rank))
        self.B = nn.Parameter(torch.randn(rank, adapted_layer.weight.size(0)))

    def forward(self, x):
        low_rank_matrix = self.A @ self.B  # 矩陣乘法
        adapted_weight = self.adapted_layer.weight + low_rank_matrix.t()  # 確保形狀正確
        return nn.functional.linear(x, adapted_weight, self.adapted_layer.bias)
