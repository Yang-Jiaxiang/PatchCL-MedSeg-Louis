import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

class Transform:
    def __init__(self, img_size, num_classes):
        self.img_size = img_size
        self.num_classes = num_classes
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        # 將 numpy array 轉換為 PIL Image 並轉換為 RGB
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert('RGB')
        image_np = np.array(image)  # 轉換為 numpy array 以檢查數據範圍
        image = self.image_transform(image)
        
        label = Image.fromarray(label)        
        # 轉換 mask
        label = self.mask_transform(label)
        
        # 將 mask 轉換為 one-hot 編碼
        label = torch.squeeze(label)  # 移除 channel 維度
        label = (label * 255).long()  # 恢復到原始像素值範圍
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).permute(2, 0, 1).float()

        return {'image': image, 'label': label}