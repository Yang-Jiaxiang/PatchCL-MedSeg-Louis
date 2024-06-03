import os
import sys
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import math
import time
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--dataset_path', type=str, default='/home/S312112021/dataset/0_data_dataset_voc_950_kidney', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='dataset/splits/kidney', help='Output directory for results')
    parser.add_argument('--patch_size', type=int, default=14, help='Batch size for contrastive learning')
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of the embedding vectors')
    parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--ContrastieWeights', type=float, default=0, help='Weight for PatchCL loss')
    parser.add_argument('--save_interval', type=int, default=2, help='Interval (in epochs) for saving the model')
    
    return parser.parse_args()

base_path = '/home/S312112021/PatchCL-MedSeg-jiyu'
voc_mask_color_map = [
    [0, 0, 0], # _background
    [128, 0, 0] # kidney
]

sys.path.append(base_path)

from utils.transform import Transform
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
from utils.datasets_PASCAL_findContours import PascalVOCDataset
from utils.queues import Embedding_Queues
from utils.CELOSS import CE_loss
from utils.patch_utils import _get_patches
from utils.aug_utils import batch_augment
from utils.get_embds import get_embeddings
from utils.plg_loss import simple_PCGJCL
from utils.torch_poly_lr_decay import PolynomialLRDecay
from utils.loss_file import save_loss
from utils_performance import DiceCoefficient, Accuracy, MeanIOU
from utils.select_reliable import select_reliable

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_bn_stats(model, train_loader):
    model.train()
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(dev)
            _ = model(imgs)

def validate(model, val_loader, criterion, num_classes):
    model.eval()
    model.module.contrast = False
    val_loss = 0.0
    total_samples = 0

    dice_coeff = DiceCoefficient(threshold=0.5)
    accuracy_metric = Accuracy(threshold=0.5)
    miou_metric = MeanIOU(threshold=0.5)

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc='Validating', leave=True):
            imgs = imgs.to(dev)
            masks = masks.to(dev)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            dice_coeff.add_batch(outputs, masks)
            accuracy_metric.add_batch(outputs, masks)
            miou_metric.add_batch(outputs, masks)
            
            val_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

    val_loss /= total_samples
    val_dice = dice_coeff.evaluate()
    val_accuracy = accuracy_metric.evaluate()
    val_miou = miou_metric.evaluate()

    return val_loss, val_miou, val_accuracy, val_dice

def get_dynamic_weight(epoch, end_epochs):
    interval = 10
    start_weight = 0.1
    max_weight = 1.0
    num_intervals = end_epochs // interval
    weight_increment = (max_weight - start_weight) / nsum_intervals

    if epoch < interval:
        weight = start_weight
    else:
        weight = start_weight + (epoch // interval) * weight_increment
    
    return weight

def train(
    model,
    teacher_model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    dev, 
    start_epochs, 
    end_epochs, 
    step_name, 
    num_classes, 
    img_size, 
    batch_size, 
    patch_size, 
    embedding_size, 
    ContrastieWeights, 
    save_interval, 
    save_loss_model_path, 
    save_loss_path
):
    embd_queues = Embedding_Queues(num_classes)

    for c_epochs in range(start_epochs,end_epochs):
        c_epochs += 1

        epoch_t_loss = 0
        total_t_supervised_loss = 0
        total_t_contrastive_loss = 0

        dice_coeff = DiceCoefficient(threshold=0.5)
        accuracy_metric = Accuracy(threshold=0.5)
        miou_metric = MeanIOU(threshold=0.5)

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {c_epochs}/{end_epochs}", unit="batch"):          
            optimizer.zero_grad()
            start_time = time.time()
            patch_list = _get_patches(
                imgs, masks,
                classes=num_classes,
                background=True,
                img_size=img_size,
                patch_size=patch_size
            )

            augmented_patch_list = batch_augment(patch_list, patch_size)
            aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
            qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

            model = model.train()
            model.module.contrast = True
            student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, batch_size)

            teacher_model.train()
            teacher_model.module.contrast = True
            teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, False, batch_size)

            embd_queues.enqueue(teacher_embedding_list)

            PCGJCL_loss = simple_PCGJCL(student_emb_list, embd_queues, embedding_size, 0.2 , 4, psi=4096)

            imgs, masks = imgs.to(dev), masks.to(dev)
            model.module.contrast = False
            out = model(imgs)
            supervised_loss = criterion(out, masks)

            dice_coeff.add_batch(out, masks)
            accuracy_metric.add_batch(out, masks)
            miou_metric.add_batch(out, masks)

            PCGJCL_loss = PCGJCL_loss.to(dev)
            
            
            if step_name == "supervised-Pretraining":
                if ContrastieWeights == 0.0:
                    PatchCL_weight = get_dynamic_weight(c_epochs, end_epochs)
                else:
                    PatchCL_weight = ContrastieWeights
            else:
                PatchCL_weight = 0.5
            
            loss = supervised_loss + PatchCL_weight * PCGJCL_loss

            total_t_contrastive_loss += PCGJCL_loss.item()
            total_t_supervised_loss += supervised_loss.item()
            epoch_t_loss += loss.item()
            
            if step_name == "supervised-Pretraining":
                loss.backward()
            else:
                loss.backward(retain_graph=True)

            optimizer.step()

            for param_stud, param_teach in zip(model.parameters(), teacher_model.parameters()):
                param_teach.data.copy_(0.001 * param_stud + 0.999 * param_teach)

            end_time = time.time()

        avg_t_epoch_loss = epoch_t_loss / len(train_loader)
        avg_t_supervised_loss = total_t_supervised_loss / len(train_loader)
        avg_t_contrastive_loss = total_t_contrastive_loss / len(train_loader)
        
        avg_t_dice = dice_coeff.evaluate()
        avg_t_accuracy = accuracy_metric.evaluate()
        avg_t_miou = miou_metric.evaluate()

        reset_bn_stats(model, train_loader)
        val_loss, val_miou, val_accuracy, val_dice = validate(model, val_loader, criterion, num_classes)

        save_loss(
            t_total_loss = f"{avg_t_epoch_loss:.4f}", 
            t_supervised_loss=f"{avg_t_supervised_loss:.4f}", 
            t_contrastive_loss=f"{avg_t_contrastive_loss:.4f}", 
            t_miou = f"{avg_t_miou:.4f}",    
            t_accuracy = f"{avg_t_accuracy:.4f}",
            t_dice = f"{avg_t_dice:.4f}",
            t_consistency_loss = 0,
            v_total_loss = f"{val_loss:.4f}", 
            v_supervised_loss = f"{val_loss:.4f}", 
            v_miou = f"{val_miou:.4f}",    
            v_accuracy = f"{val_accuracy:.4f}",
            v_dice = f"{val_dice:.4f}",
            PatchCL_weight = PatchCL_weight,
            filename= f'{save_loss_path}_{step_name}.csv'
        )

        if (c_epochs) % save_interval == 0:
            os.makedirs(save_loss_model_path, exist_ok=True)
            torch.save(model, f"{save_loss_model_path}/model_{step_name}_{c_epochs}-s.pth")
    
    return model, teacher_model

def load_pretrained_model(model, teacher_model, save_model_path, epoch):
    model_path = f"{save_model_path}{epoch}-s.pth"
    teacher_model_path = f"{save_model_path}{epoch-10}-s.pth"

    model = torch.load(model_path)
    teacher_model = torch.load(teacher_model_path)

    model.eval()
    teacher_model.eval()
    return model, teacher_model

def to_one_hot(tensor, num_classes):
    n, h, w = tensor.shape
    one_hot = torch.zeros(n, num_classes, h, w).to(tensor.device)
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot



# +
def main():
    args = parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    patch_size = args.patch_size
    embedding_size = args.embedding_size
    img_size = args.img_size
    batch_size = args.batch_size
    num_classes = args.num_classes
    ContrastieWeights = args.ContrastieWeights
    save_interval = args.save_interval

    save_loss_path = f'output/loss_{patch_size}-{ContrastieWeights}'
    save_loss_model_path = f'output/{patch_size}-{ContrastieWeights}'
    
    cross_entropy_loss = CE_loss(num_classes, image_size=img_size)

    model = Network(num_classes, embedding_size=embedding_size)
    teacher_model = Network(num_classes, embedding_size=embedding_size)

    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model.load_state_dict(model.state_dict())

    model = nn.DataParallel(model)
    model = model.to(dev)
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = teacher_model.to(dev)

    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_ssl = torch.optim.SGD(model.parameters(), lr=0.007)
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

    print('\n\n\n================> Total stage 1/7: Supervised training on labeled images (SupOnly)')
    supervised_start_epoch = 0
    supervised_end_epoch = 100
    
    model, teacher_model = train(
        model, 
        teacher_model, 
        train_loader,
        val_loader, 
        optimizer_pretrain, 
        cross_entropy_loss, 
        dev, 
        supervised_start_epoch, 
        supervised_end_epoch, 
        "supervised-Pretraining", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        ContrastieWeights,
        save_interval,
        save_loss_model_path,
        save_loss_path
    )

    print('\n\n\n================> Total stage 2/7: Select reliable images for the 1st stage re-training')
    save_model_path = f"{save_loss_model_path}/model_supervised-Pretraining_"
    model, teacher_model = load_pretrained_model(model, teacher_model, save_model_path, supervised_end_epoch)
    reliable_dataset, remaining_dataset= select_reliable(model, teacher_model, unlabeled_loader, num_classes)
    print('reliable_dataset:', len(reliable_dataset))
    print('remaining_dataset:', len(remaining_dataset))

    print('\n\n\n================> Total stage 3/7: Concat train_dataset remaining_dataset')
    combined_dataset = ConcatDataset([train_dataset, reliable_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print('\n\n\n================> Total stage 4/7: Semi-supervised training with reliable images (SSL)')
    SSL_start_epoch = 0
    SSL_end_epoch = 100
    
    model, teacher_model = train(
        model, 
        teacher_model, 
        train_loader,
        val_loader, 
        optimizer_pretrain, 
        cross_entropy_loss, 
        dev, 
        SSL_start_epoch, 
        SSL_end_epoch, 
        "SSL-reliable-st1", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        ContrastieWeights,
        save_interval,
        save_loss_model_path,
        save_loss_path
    )
    
    print('\n\n\n================> Total stage 5/7: Generate pseudo labels for remaining images')

    if remaining_dataset is None:
        return

    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    save_model_path = f"{save_loss_model_path}/model_SSL-reliable-st1_{SSL_end_epoch}"
    model, teacher_model = load_pretrained_model(model, teacher_model, save_model_path, SSL_step1_epoch)
    print('remaining_dataset:', len(remaining_dataset))

    print('\n\n\n================> Total stage 6/7: Concat train_dataset remaining_dataset')
    combined_dataset = ConcatDataset([train_dataset, remaining_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print('\n\n\n================> Total stage 7/7: Semi-supervised training with reliable images (SSL)')
    SSL_step2_start_epoch = 0
    SSL_step2_end_epoch = 100
    
    model, teacher_model = train(
        model, 
        teacher_model, 
        train_loader,
        val_loader, 
        optimizer_pretrain, 
        cross_entropy_loss, 
        dev, 
        SSL_step2_start_epoch, 
        SSL_step2_end_epoch, 
        "SSL-reliable-st2", 
        num_classes, 
        img_size, 
        batch_size, 
        patch_size, 
        embedding_size,
        ContrastieWeights,
        save_interval,
        save_loss_model_path,
        save_loss_path
    )
    
    print('\n\n\n================> Finish')


# -

if __name__ == '__main__':
    main()
