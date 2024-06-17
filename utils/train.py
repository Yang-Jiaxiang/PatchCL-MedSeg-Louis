import os
import torch
import torch.nn as nn
import math
import time
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
import numpy as np
from tqdm import tqdm

from utils.queues import Embedding_Queues
from utils.patch_utils import _get_patches
from utils.aug_utils import batch_augment
from utils.get_embds import get_embeddings
from utils.plg_loss import simple_PCGJCL
from utils.loss_file import save_loss
from utils.performance import DiceCoefficient, MeanIOU
from utils.validate import validate
from utils.update_ema_variables import update_ema_variables
from utils.reset_bn_stats import reset_bn_stats

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    model,
    teacher_model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    end_epochs, 
    step_name, 
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
):
    embd_queues = Embedding_Queues(num_classes)
    
    for c_epochs in range(0,end_epochs):
        c_epochs += 1

        epoch_t_loss = 0
        total_t_supervised_loss = 0
        total_t_contrastive_loss = 0

        dice_coeff = DiceCoefficient()
        miou_metric = MeanIOU()

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
            teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, True, batch_size)

            embd_queues.enqueue(teacher_embedding_list)

            PCGJCL_loss = simple_PCGJCL(student_emb_list, embd_queues, embedding_size, 0.2 , 4, psi=4096)

            imgs, masks = imgs.to(dev), masks.to(dev)
            model.module.contrast = False
            out = model(imgs)
            
            out_selected = out[:, 1:, :, :]  # shape: (16, C-1, 224, 224)
            masks_selected = masks[:, 1:, :, :]  # shape: (16, C-1, 224, 224)

            supervised_loss = criterion(out_selected, masks_selected)

            dice_coeff.add_batch(out_selected, masks_selected)
            miou_metric.add_batch(out_selected, masks_selected)

            PCGJCL_loss = PCGJCL_loss.to(dev)
                        
            loss = supervised_loss + contrastiveWeights * PCGJCL_loss

            total_t_contrastive_loss += PCGJCL_loss.item()
            total_t_supervised_loss += supervised_loss.item()
            epoch_t_loss += loss.item()
            
            if step_name =='supervised-Pretraining':
                loss.backward()
            else:
                loss.backward(retain_graph=True)
                
            optimizer.step()

            update_ema_variables(model, teacher_model, alpha=ema_alpha, global_step=c_epochs)
            
            end_time = time.time()

        avg_t_epoch_loss = epoch_t_loss / len(train_loader)
        avg_t_supervised_loss = total_t_supervised_loss / len(train_loader)
        avg_t_contrastive_loss = total_t_contrastive_loss / len(train_loader)
        
        avg_t_dice = dice_coeff.evaluate()
        avg_t_miou = miou_metric.evaluate()

        reset_bn_stats(model, train_loader)
        val_loss, val_miou, val_dice = validate(model, val_loader, criterion, num_classes)
    
        save_loss(
            t_total_loss = f"{avg_t_epoch_loss:.4f}", 
            t_supervised_loss=f"{avg_t_supervised_loss:.4f}", 
            t_contrastive_loss=f"{avg_t_contrastive_loss:.4f}", 
            t_miou = f"{avg_t_miou:.4f}",    
            t_dice = f"{avg_t_dice:.4f}",
            v_total_loss = f"{val_loss:.4f}", 
            v_supervised_loss = f"{val_loss:.4f}", 
            v_miou = f"{val_miou:.4f}",    
            v_dice = f"{val_dice:.4f}",
            PatchCL_weight = contrastiveWeights,
            filename= f'{save_loss_path}_{step_name}.csv'
        )

        if (c_epochs) % save_interval == 0:
            os.makedirs(save_model_path, exist_ok=True)
            torch.save(model, f"{save_model_path}/model_{step_name}_{c_epochs}-s.pth")
            torch.save(teacher_model, f"{save_model_path}/model_{step_name}_{c_epochs}-t.pth")
    
    return model, teacher_model