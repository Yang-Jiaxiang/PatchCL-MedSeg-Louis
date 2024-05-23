import os
import cv2
import sys
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
from dataloaders.utils import encode_segmap


# +
class PascalVOCDataset(Dataset):
    def __init__(self, txt_file, image_size, root_dir, transform=None, labeled=True, colormap=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.labeled = labeled
        self.colormap = colormap

        with open(txt_file, 'r') as file:
            self.image_mask_pairs = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
        height, width, _ = image.shape
        image = image[:, :height]
        
#         image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = torch.tensor(image).permute(2, 0, 1).float()

        if self.labeled:
            mask_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][1])
            mask = cv2.imread(mask_name)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            height, width, _ = mask.shape
            mask  = mask[:, :height]
#             mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = encode_segmap(mask, self.colormap)

            
        else:
            mask = None
            
        if self.labeled:
            return image, mask
        else:
            return image
