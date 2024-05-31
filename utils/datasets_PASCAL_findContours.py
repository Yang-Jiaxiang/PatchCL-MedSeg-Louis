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
def _findContours(image, threshold):
    src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgh,imgw = src.shape
    
    # src: 輸入的灰度圖像。
    # threshold_value: 閾值。如果像素值大於這個閾值，則設置為 max_value，否則設置為 0。
    # max_value: 當像素值超過閾值時分配的值。通常設置為 255，表示白色。
    # threshold_type: 閾值類型。cv2.THRESH_BINARY 表示使用二值化處理。
    gray = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # gray: 輸入的二值化圖像。在此例中，它是由 cv2.threshold 函數生成的。
    # cv2.RETR_EXTERNAL: 輪廓檢索模式。cv2.RETR_EXTERNAL 只檢測最外層的輪廓。這意味著，如果有多個嵌套的輪廓（如一個輪廓在另一個輪廓內部），只有最外層的輪廓會被檢測到。
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    maxm=0
    #fig = d2l.plt.imshow(src, cmap='gray')
    for i in range(0, len(contours)):

        # cv2.boundingRect: 透過輪廓找到外接矩形
        # 輸出：(x, y)矩形左上角座標、w 矩形寬(x軸方向)、h 矩形高(y軸方向)
        x, y, w, h = cv2.boundingRect(contours[i])
        if w<imgw and h<imgh and w+h>maxm:
            maxm=w+h
            index=i
        # 在原影像上繪製出矩形
        '''fig.axes.add_patch(d2l.plt.Rectangle((x, y), w, h, fill=False,
                           linestyle="-", edgecolor=color,
                           linewidth=2))'''
    x, y, w, h = cv2.boundingRect(contours[index])
    
    return x, y, w, h

def _cropImage(image, x, y, w, h):
    _image = image.copy()
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


# -

class PascalVOCDataset(Dataset):
    def __init__(self, txt_file, image_size, root_dir, transform=None, labeled=True, colormap=None, findContours_threshold_value = 30):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.labeled = labeled
        self.findContours_threshold_value = findContours_threshold_value
        self.colormap = colormap
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # 將圖像轉為 [0, 1] 的 tensor
        ])

        with open(txt_file, 'r') as file:
            self.image_mask_pairs = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][0])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    
        # cs2.findContours 裁切，回傳 x, y, w, h
        x, y, w, h = _findContours(image, threshold=self.findContours_threshold_value)
        cropped_image = _cropImage(image, x, y, w, h)
        
        cropped_image = cv2.resize(cropped_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = self.image_transform(cropped_image)
        
        if self.labeled:
            mask_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][1])
            mask = cv2.imread(mask_name)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            cropped_mask = _cropImage(mask, x, y, w, h)
            cropped_mask = cv2.resize(cropped_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = encode_segmap(cropped_mask, self.colormap) # output (244,244) 
            
            one_hot_mask = np.zeros((2, self.image_size, self.image_size), dtype=np.uint8)
            one_hot_mask[0, mask == 0] = 1
            one_hot_mask[1, mask == 1] = 1
            mask = one_hot_mask # (2, 244, 244)
            mask = torch.from_numpy(mask).float()
            
        else:
            mask = None
            
        if self.labeled:
            return image, mask
        else:
            return image
