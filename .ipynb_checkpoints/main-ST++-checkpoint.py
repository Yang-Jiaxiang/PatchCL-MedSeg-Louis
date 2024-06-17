import os
import json
import torch
import torch.nn as nn
import math
import time
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
import numpy as np
import argparse

from utils.transform import Transform
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
from utils.datasets_PASCAL import PascalVOCDataset
from utils.DISLOSS import DiceLoss
from utils.torch_poly_lr_decay import PolynomialLRDecay
from utils.select_reliable import select_reliable, Label
from utils.load_pretrained_model import load_pretrained_model
from utils.train import train


with open('voc_mask_color_map.json', 'r') as file:
    JsonData = json.load(file)
voc_mask_color_map = JsonData['voc_mask_color_map']

supervised_epoch = 100
SSL_step1_epoch = 100
SSL_step2_epoch = 100
process = 'ST++'

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--dataset_path', type=str, default='/home/S312112021/dataset/0_data_dataset_voc_950_kidney', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='dataset/splits/kidney', help='Output directory for results')
    parser.add_argument('--patch_size', type=int, default=14, help='Batch size for contrastive learning')
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of the embedding vectors')
    parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--contrastiveWeights', type=float, default=0, help='Weight for PatchCL loss')
    parser.add_argument('--save_interval', type=int, default=2, help='Interval (in epochs) for saving the model')
    parser.add_argument('--ema_alpha', type=float, default=0, help='Weight for EMA')
    return parser.parse_args()
    
# +
def main():
    args = parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    patch_size = args.patch_size
    embedding_size = args.embedding_size
    img_size = args.img_size
    batch_size = args.batch_size
    num_classes = len(voc_mask_color_map)
    contrastiveWeights = args.contrastiveWeights
    save_interval = args.save_interval
    ema_alpha = args.ema_alpha

    save_loss_path = f'output/{process}/loss_{patch_size}-{contrastiveWeights}'
    save_model_path = f'output/{process}/{patch_size}-{contrastiveWeights}'
    
    loss_function = DiceLoss() 
    
    model = Network(num_classes, embedding_size=embedding_size)
    teacher_model = Network(num_classes, embedding_size=embedding_size)

    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model.load_state_dict(model.state_dict())

    model = nn.DataParallel(model)
    model = model.to(dev)
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = teacher_model.to(dev)

    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=0.001) # , weight_decay=1e-4
    optimizer_ssl = torch.optim.SGD(model.parameters(), lr=0.007) # , weight_decay=1e-4
    scheduler = PolynomialLRDecay(optimizer=optimizer_pretrain, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)

    labeled_dataset = PascalVOCDataset(txt_file=output_dir + "/1-3/labeled.txt", image_size=img_size, root_dir=dataset_path, labeled=True, colormap=voc_mask_color_map)
    labeled_dataset_length = len(labeled_dataset)

    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    train_length = int(train_ratio * labeled_dataset_length)
    val_length = labeled_dataset_length - train_length

    train_dataset, val_dataset = random_split(labeled_dataset, [train_length, val_length])
    unlabeled_dataset = PascalVOCDataset(txt_file=output_dir + "/1-3/unlabeled.txt", image_size=img_size, root_dir=dataset_path, labeled=False, colormap=voc_mask_color_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print('number of train_dataset: ', len(train_dataset))
    print('number of val_dataset: ', len(val_dataset))
    print('number of unlabeled_dataset: ', len(unlabeled_dataset))

    print('\n\n\n================> Total stage 1/6: Supervised training on labeled images (SupOnly)')    
    model, teacher_model = train(
        model, 
        teacher_model, 
        train_loader,
        val_loader, 
        optimizer_pretrain, 
        loss_function, 
        supervised_epoch, 
        "supervised-Pretraining", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        contrastiveWeights,
        ema_alpha,
        save_interval,
        save_model_path,
        save_loss_path
    )

    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    model, teacher_model = load_pretrained_model(model, teacher_model, f"{save_model_path}/model_supervised-Pretraining_", supervised_epoch)
    reliable_dataset, remaining_dataset= select_reliable(model, teacher_model, unlabeled_loader, num_classes)
    print('reliable_dataset:', len(reliable_dataset))
    print('remaining_dataset:', len(remaining_dataset))
    
    print('\n\n\n================> Total stage 3/6: Concat train_dataset reliable_dataset')
    combined_dataset = ConcatDataset([train_dataset, reliable_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("combined_dataset: ", len(combined_dataset))
    
    print('\n\n\n================> Total stage 4/6: Semi-supervised training with reliable images (SSL)')
    
    model, teacher_model = train(
        model, 
        teacher_model, 
        combined_loader,
        val_loader, 
        optimizer_pretrain, 
        loss_function, 
        SSL_step1_epoch, 
        "SSL-reliable-st1", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        contrastiveWeights,
        ema_alpha,
        save_interval,
        save_model_path,
        save_loss_path
    )
    
    print('\n\n\n================> Total stage 5/6: Generate pseudo labels for remaining images')

    if len(remaining_dataset) < batch_size:
        print("remaining_dataset < batch size")
    
    model, teacher_model = load_pretrained_model(model, teacher_model, f"{save_model_path}/model_SSL-reliable-st1_", SSL_step1_epoch)
    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    remaining_label_dataset = Label(model, remaining_loader, num_classes, device=dev)
    remaining_label_loader = DataLoader(remaining_label_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print("remaining_label_loader: ", len(remaining_label_loader))
    
    if len(remaining_label_loader) < 2:
        print("remaining_label_loader < 2")
        return
    
    print('\n\n\n================> Total stage 6/6: Semi-supervised training with reliable images (SSL)')
    
    model, teacher_model = train(
        model, 
        teacher_model, 
        remaining_label_loader,
        val_loader, 
        optimizer_pretrain, 
        loss_function, 
        SSL_step2_epoch, 
        "SSL-reliable-st2", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        contrastiveWeights,
        ema_alpha,
        save_interval,
        save_model_path,
        save_loss_path
    )
    
    print('\n\n\n================> Finish')


if __name__ == '__main__':
    main()
