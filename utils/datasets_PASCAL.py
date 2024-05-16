import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


class PascalVOCDataset(Dataset):
    def __init__(self, txt_file, image_size, root_dir, transform=None, labeled=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.labeled = labeled

        with open(txt_file, 'r') as file:
            self.image_mask_pairs = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = image.permute(0, 3, 1, 2)

        if self.labeled:
            mask_name = os.path.join(self.root_dir, self.image_mask_pairs[idx][1])
            mask = cv2.imread(mask_name)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = mask.permute(0, 3, 1, 2)
        else:
            mask = None
            
        if self.labeled:
            return image, mask
        else:
            return image
    
    
    
    
    
# class LabData(Dataset):
#     def __init__(
#         self,
#         base_dir=None,
#         split="train",
#         num=None,
#         transform=None,
#         ops_weak=None,
#         ops_strong=None,
#         labeled_slice=None,
#     ):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.transform = transform
#         self.ops_weak = ops_weak
#         self.ops_strong = ops_strong

#         if self.split == "train":
#             with open(self._base_dir + "/train_slices.list", "r") as f1:
#                 self.sample_list = f1.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list[0:labeled_slice]]

#         elif self.split == "val":
#             with open(self._base_dir + "/val.list", "r") as f:
#                 self.sample_list = f.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#         if num is not None and self.split == "train":
#             self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train":
#             h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
#         else:
#             h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
#         image = h5f["image"][:]
#         label = h5f["label"][:]
#         sample = {"image": image, "label": label}
#         if self.split == "train":
#             if None not in (self.ops_weak, self.ops_strong):
#                 sample = self.transform(sample, self.ops_weak, self.ops_strong)
#             else:
#                 sample = self.transform(sample)
#         sample["idx"] = idx
#         return sample

# class UnlabData(Dataset):
#     def __init__(
#         self,
#         base_dir=None,
#         split="train",
#         num=None,
#         transform=None,
#         ops_weak=None,
#         ops_strong=None,
#         labeled_slice=None,
#     ):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.transform = transform
#         self.ops_weak = ops_weak
#         self.ops_strong = ops_strong

#         assert bool(ops_weak) == bool(
#             ops_strong
#         ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

#         if self.split == "train":
#             with open(self._base_dir + "/train_slices.list", "r") as f1:
#                 self.sample_list = f1.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list[0:labeled_slice]]

#         elif self.split == "val":
#             with open(self._base_dir + "/val.list", "r") as f:
#                 self.sample_list = f.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#         if num is not None and self.split == "train":
#             self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train":
#             h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
#         else:
#             h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
#         image = h5f["image"][:]
#         label = h5f["label"][:]
#         sample = {"image": image, "label": label}
#         if self.split == "train":
#             if None not in (self.ops_weak, self.ops_strong):
#                 sample = self.transform(sample, self.ops_weak, self.ops_strong)
#             else:
#                 sample = self.transform(sample)
#         sample["idx"] = idx
#         return sample