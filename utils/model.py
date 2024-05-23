import torch

import torch.nn as nn
import segmentation_models_pytorch as smp

class Network(nn.Module):
    def __init__(self, num_classes, embedding_size=64):
        super(Network, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(
            'resnet18', 
            classes=num_classes, 
            in_channels=3, 
            encoder_weights='imagenet', 
            activation=None
        )
        self.encoder = self.seg_model.encoder
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Change the output size of avgpool to (1, 1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),  # Adjust input size to match the new output of avgpool
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_size),
            nn.InstanceNorm1d(embedding_size)
        )
        self.contrast = False

    def forward(self,x):
        if self.contrast is True:
            x =self.encoder(x)
            x = x[-1]  # Taking the last feature map only
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x =self.fc(x)
            return x
        else:
            x = self.seg_model(x)
#             x = torch.sigmoid(x)  # Apply sigmoid to the output of seg_model
            return x

# +
# class Network(nn.Module):

#   def __init__(self, num_classes, embedding_size=64):
#     super(Network,self).__init__()
#     self.seg_model = smp.DeepLabV3Plus('resnet50',classes=num_classes,in_channels=3,encoder_weights='imagenet',activation=None)
#     self.encoder= self.seg_model.encoder
#     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#     self.fc = nn.Sequential(
#                 nn.Linear(2048,2048),
#                 nn.InstanceNorm1d(2048),
#                 nn.ReLU(),
#                 nn.Linear(2048,embedding_size),
#                 nn.InstanceNorm1d(embedding_size)
#         ) #2048 for ResNet50 and 101;
#     self.contrast=False

#   def forward(self,x):
#     if self.contrast is True:
#         x =self.encoder(x)
#         x=x[-1] #Taking the last feature map only
#         x =self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x =self.fc(x)
#         return x
#     else:
#         return self.seg_model(x)
