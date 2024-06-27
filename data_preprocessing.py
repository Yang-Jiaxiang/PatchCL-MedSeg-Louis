import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--dataset_path', type=str, default='/home/S312112021/dataset/0_data_dataset_voc_950', help='Path to the dataset')
    parser.add_argument('--voc_output_dir', type=str, default='dataset/splits/kidney', help='Output directory for results')
    parser.add_argument('--voc_splits', type=str, default='1-3', help='splits')
    parser.add_argument('--crop_output_dir', type=str, default='data/0_data_dataset_voc_950', help='crop_output_dir')
    parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation set')
    return parser.parse_args()

def _findContours(image, threshold):
    gray = ImageOps.grayscale(image)
    binary = gray.point(lambda p: p > threshold and 255)
    contours, _ = cv2.findContours(np.array(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxm = 0
    index = -1
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < image.width and h < image.height and w + h > maxm:
            maxm = w + h
            index = i

    if index == -1:
        return None

    x, y, w, h = cv2.boundingRect(contours[index])
    
    return x, y, w, h

def _cropImage(image, x, y, w, h):
    cropped_image = image.crop((x, y, x + w, y + h))
    return cropped_image

def labeled(id_path, dataset_path, crop_output_dir, img_size):
    with open(id_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        img_path, mask_path = line.strip().split()
        og_img_path = os.path.join(dataset_path, img_path)
        mask_img_path = os.path.join(dataset_path, mask_path)
        image = Image.open(og_img_path)
        mask = Image.open(mask_img_path)
        
        contour = _findContours(image, 30)
        if contour is not None:
            x, y, w, h = contour
            cropped_image = _cropImage(image, x, y, w, h)
            cropped_mask = _cropImage(mask, x, y, w, h)
            cropped_image = cropped_image.resize((img_size, img_size))
            cropped_mask = cropped_mask.resize((img_size, img_size))
            
            save_image_path = os.path.join(crop_output_dir, img_path)
            save_mask_path = os.path.join(crop_output_dir, mask_path)
            
            save_image_dir = os.path.dirname(save_image_path)
            save_mask_dir = os.path.dirname(save_mask_path)
            if not os.path.exists(save_image_dir):
                os.makedirs(save_image_dir)
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
                
            cropped_image.save(save_image_path)
            cropped_mask.save(save_mask_path)
            print(f'Cropped image saved to {save_image_path}')
            print(f'Cropped mask saved to {save_mask_path}')
        else:
            print(f'No contours found in image {img_path}')

def unlabeled(id_path, dataset_path, crop_output_dir, img_size):
    with open(id_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        img_path = line.strip()
        og_img_path = os.path.join(dataset_path, img_path)
        image = Image.open(og_img_path)
        
        contour = _findContours(image, 30)
        if contour is not None:
            x, y, w, h = contour
            cropped_image = _cropImage(image, x, y, w, h)
            cropped_image = cropped_image.resize((img_size, img_size))
            
            save_image_path = os.path.join(crop_output_dir, img_path)
            
            save_image_dir = os.path.dirname(save_image_path)
            if not os.path.exists(save_image_dir):
                os.makedirs(save_image_dir)
                
            cropped_image.save(save_image_path)
            print(f'Cropped image saved to {save_image_path}')
        else:
            print(f'No contours found in image {img_path}')

def split_labeled_data(labeled_id_path, train_output_path, val_output_path, val_ratio=0.1):
    with open(labeled_id_path, 'r') as f:
        labeled_ids = f.read().splitlines()

    total_size = len(labeled_ids)
    val_size = int(total_size * val_ratio)

    random.shuffle(labeled_ids)
    val_ids = labeled_ids[:val_size]
    train_ids = labeled_ids[val_size:]

    with open(train_output_path, 'w') as f:
        for id in train_ids:
            f.write(f"{id}\n")

    with open(val_output_path, 'w') as f:
        for id in val_ids:
            f.write(f"{id}\n")

    print(f"總數據量: {total_size}")
    print(f"訓練集大小: {len(train_ids)}")
    print(f"驗證集大小: {len(val_ids)}")
    
def main():
    args = parse_args()
    dataset_path = args.dataset_path
    voc_output_dir = args.voc_output_dir
    voc_splits = args.voc_splits
    crop_output_dir = args.crop_output_dir
    img_size = args.img_size
    val_ratio = args.val_ratio
    
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)
        
    val_id_path = f'{voc_output_dir}/val.txt'
    label_id_path = f'{voc_output_dir}/{voc_splits}/labeled.txt'
    unlabel_id_path = f'{voc_output_dir}/{voc_splits}/unlabeled.txt'
    
    labeled(val_id_path, dataset_path, crop_output_dir, img_size)
    labeled(label_id_path, dataset_path, crop_output_dir, img_size)
    labeled(unlabel_id_path, dataset_path, crop_output_dir, img_size)
    
    train_output_path = f'{voc_output_dir}/{voc_splits}/train.txt'
    val_output_path = f'{voc_output_dir}/{voc_splits}/val.txt'

    split_labeled_data(label_id_path, train_output_path, val_output_path, val_ratio)

if __name__ == '__main__':
    main()
