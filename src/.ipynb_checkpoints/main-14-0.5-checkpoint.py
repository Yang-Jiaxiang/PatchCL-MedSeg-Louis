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
from torchvision import transforms


base_path = '/home/u5169119/PatchCL-MedSeg-jiyu'
dataset_path = '/home/u5169119/dataset/0_data_dataset_voc_950_kidney'
output_dir = 'dataset/splits/kidney'

contrastive_batch_size = 14
embedding_size = 128
img_size = 224
batch_size = 16
num_classes = 2
ContrastieWeights = 0.5 # PatchCL loss weight
save_interval = 2  # 每 10 輪儲存一次

parameter = f'Resnet18_Image-{img_size}_patchSize-{contrastive_batch_size}_ContrastieWeights-{ContrastieWeights}'
save_loss_path = f'output/loss_{contrastive_batch_size}-{ContrastieWeights}'
save_loss_model_path = f'output/{contrastive_batch_size}-{ContrastieWeights}'


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
from utils.const_reg import consistency_cost
from utils.plg_loss import PCGJCL, simple_PCGJCL
from utils.torch_poly_lr_decay import PolynomialLRDecay
from utils.loss_file import save_loss
from utils_performance import DiceCoefficient, Accuracy, MeanIOU

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_bn_stats(model, train_loader):
    model.train()
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(dev)
            _ = model(imgs)

def validate(model, val_loader, criterion, num_classes):
    model.eval()  # 設置模型為評估模式
    model.module.contrast = False
    val_loss = 0.0
    total_samples = 0

    # 初始化指標計算實例
    dice_coeff = DiceCoefficient(threshold=0.5)
    accuracy_metric = Accuracy(threshold=0.5)
    miou_metric = MeanIOU(threshold=0.5)

    with torch.no_grad():  # 禁用梯度計算
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

def train(model, teacher_model, train_loader, val_loader, optimizer, criterion, dev, start_epochs, end_epochs, step_name, num_classes, img_size, contrastive_batch_size, ContrastieWeights, save_loss_path):
    embd_queues = Embedding_Queues(num_classes)

    for c_epochs in range(start_epochs,end_epochs):
        c_epochs += 1

        epoch_t_loss = 0
        total_t_supervised_loss = 0
        total_t_contrastive_loss = 0

        # 初始化指標計算實例
        dice_coeff = DiceCoefficient(threshold=0.5)
        accuracy_metric = Accuracy(threshold=0.5)
        miou_metric = MeanIOU(threshold=0.5)

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {c_epochs}/{end_epochs}", unit="batch"):
            optimizer.zero_grad()
            start_time = time.time()  # step 開始時間

            patch_list = _get_patches(
                imgs, masks,
                classes=num_classes,
                background=True,
                img_size=img_size,
                patch_size=contrastive_batch_size
            )

            augmented_patch_list = batch_augment(patch_list, contrastive_batch_size)
            # Convert to tensor
            aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
            qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

            # Get embeddings of qualified patches through student model
            model = model.train()
            model.module.contrast = True
            student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True)

            # Get embeddings of augmented patches through teacher model
            teacher_model.train()
            teacher_model.module.contrast = True
            teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, False)

            # Enqueue these
            embd_queues.enqueue(teacher_embedding_list)

            # Calculate PCGJCL loss
            PCGJCL_loss = simple_PCGJCL(student_emb_list, embd_queues, embedding_size, 0.2 , 4, psi=4096)    

            # Calculate supervised loss
            imgs, masks = imgs.to(dev), masks.to(dev)
            model.module.contrast = False
            out = model(imgs)
            supervised_loss = criterion(out, masks)

            dice_coeff.add_batch(out, masks)
            accuracy_metric.add_batch(out, masks)
            miou_metric.add_batch(out, masks)

            # Total loss
            PCGJCL_loss = PCGJCL_loss.to(dev)
            loss = supervised_loss + ContrastieWeights * PCGJCL_loss

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
            v_contrastive_loss=0, 
            v_consistency_loss=0, 
            filename= f'{save_loss_path}_{step_name}.csv'
        )

        if (c_epochs) % save_interval == 0:
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


def select_reliable(model, teacher_model, data_loader, num_classes, threshold=0.1, device='cuda'):
    criterion = torch.nn.functional.cross_entropy

    model.eval()
    teacher_model.eval()
    
    model.module.contrast = False
    teacher_model.module.contrast = False

    reliable_images = []
    reliable_outputs = []
    remaining_images = []

    tbar = tqdm(data_loader)

    with torch.no_grad():
        for imgs in tbar:
            imgs = imgs.to(device)

            student_outputs = model(imgs)
            teacher_outputs = teacher_model(imgs)

            # Consistency loss
            teacher_labels = teacher_outputs.argmax(dim=1)
            consistency_loss = criterion(student_outputs, teacher_labels, reduction='none')

            # Convert outputs to (b, c, w, h) with one-hot encoding
            student_outputs_one_hot = torch.nn.functional.one_hot(student_outputs.argmax(dim=1), num_classes=num_classes)
            student_outputs_one_hot = student_outputs_one_hot.permute(0, 3, 1, 2).float()

            for img, output, loss in zip(imgs, student_outputs_one_hot, consistency_loss):
                if loss.mean().item() < threshold:
                    reliable_images.append(img.cpu().numpy())
                    reliable_outputs.append(output.cpu().numpy())
                else:
                    remaining_images.append(img.cpu().numpy())
    #
    #  Convert lists to tensors
    reliable_images_tensor = torch.tensor(reliable_images)
    reliable_outputs_tensor = torch.tensor(reliable_outputs)
    # Create reliable dataset
    reliable_dataset = TensorDataset(reliable_images_tensor, reliable_outputs_tensor)
    remaining_dataset = TensorDataset(torch.tensor(remaining_images))
    return reliable_dataset, remaining_dataset


def main():
    cross_entropy_loss = CE_loss(num_classes, image_size=img_size)

    model = Network(num_classes, embedding_size=embedding_size)
    teacher_model = Network(num_classes, embedding_size=embedding_size)

    # Turning off gradients for teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    # Ensuring both models have same weight
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

    # 設定 train 和 val 的比例
    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    # 計算 train 和 val 的長度
    train_length = int(train_ratio * labeled_dataset_length)
    val_length = labeled_dataset_length - train_length

    train_dataset, val_dataset = random_split(labeled_dataset, [train_length, val_length])
    unlabeled_dataset = PascalVOCDataset(txt_file=output_dir + "/1-3/unlabeled.txt", image_size=img_size, root_dir=dataset_path, labeled=False, colormap=voc_mask_color_map)

    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 打印資料集的大小
    print('number of train_dataset: ', len(train_dataset))
    print('number of val_dataset: ', len(val_dataset))
    print('number of unlabeled_dataset: ', len(unlabeled_dataset))

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n\n\n================> Total stage 1/7: Supervised training on labeled images (SupOnly)')
    supervised_start_epoch = 0
    supervised_end_epoch = 100
#     model, teacher_model = train(model, teacher_model, train_loader, val_loader, optimizer_pretrain, cross_entropy_loss, dev, supervised_start_epoch, supervised_end_epoch, "supervised-Pretraining", num_classes, img_size, contrastive_batch_size, ContrastieWeights, save_loss_path)

    # <====================== Sgenerate pseudo labels ======================>
    print('\n\n\n================> Total stage 2/7: Select reliable images for the 1st stage re-training')

    save_model_path = f"{save_loss_model_path}/model_supervised-Pretraining_"
    # 重新加載模型
    model, teacher_model = load_pretrained_model(model, teacher_model, save_model_path, supervised_end_epoch)

    # 篩選可靠的圖像和標籤
    reliable_dataset, remaining_dataset= select_reliable(model, teacher_model, unlabeled_loader, num_classes)
    
    print('reliable_dataset:', len(reliable_dataset))
    print('remaining_dataset:', len(remaining_dataset))
    # <================================ Concat dataset =================================>
    print('\n\n\n================> Total stage 3/7: Concat train_dataset remaining_dataset')
    # 合併 train_dataset 和 reliable_dataset
    combined_dataset = ConcatDataset([train_dataset, reliable_dataset])

    # 使用新的 combined_dataset 創建新的 DataLoader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # <====================== Semi-supervised training with reliable images (SSL) ======================>
    print('\n\n\n================> Total stage 4/7: Semi-supervised training with reliable images (SSL)')
    SSL_start_epoch = 0
    SSL_end_epoch = 100
    model, teacher_model = train(model, teacher_model, combined_loader, val_loader, optimizer_ssl, cross_entropy_loss, dev, SSL_start_epoch, SSL_end_epoch, "SSL-reliable-st1", num_classes, img_size, contrastive_batch_size, ContrastieWeights, save_loss_path)

    # <====================== Generate pseudo labels for remaining images ======================>
    print('\n\n\n================> Total stage 5/7: Generate pseudo labels for remaining images')

    # 如果沒有剩餘的圖像，則跳出 main function 
    if remaining_dataset is None:
        return

    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    

    save_model_path = f"{save_loss_model_path}/model_SSL-reliable-st1_{epoch}"
    # 重新加載模型
    model, teacher_model = load_pretrained_model(model, teacher_model, save_model_path, SSL_step1_epoch)

    # 篩選可靠的圖像和標籤
    reliable_dataset, remaining_dataset = select_reliable(model, teacher_model, remaining_loader, num_classes)

    # <================================ Concat dataset =================================>
    print('\n\n\n================> Total stage 6/7: Concat train_dataset remaining_dataset')
    # 合併 train_dataset 和 reliable_dataset
    combined_dataset = ConcatDataset([train_dataset, reliable_dataset])

    # 使用新的 combined_dataset 創建新的 DataLoader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # <====================== Semi-supervised training with reliable images (SSL) ======================>
    print('\n\n\n================> Total stage 7/7: Semi-supervised training with reliable images (SSL)')
    SSL_step2_start_epoch = 0
    SSL_step2_end_epoch = 100
    model, teacher_model = train(model, teacher_model, combined_loader, val_loader, optimizer_ssl, cross_entropy_loss, dev, SSL_step2_start_epoch, SSL_step2_end_epoch, "SSL-remaining-st2", num_classes, img_size, contrastive_batch_size, ContrastieWeights, save_loss_path)


if __name__ == '__main__':
    main()
