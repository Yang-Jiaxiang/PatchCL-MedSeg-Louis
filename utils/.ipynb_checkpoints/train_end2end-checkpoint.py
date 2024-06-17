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
from utils.select_reliable import consistency_regularization_CELoss

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_end2end(
    model,
    teacher_model, 
    train_loader, 
    unlabeled_loader,
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
        total_t_consistency_loss = 0 

        dice_coeff = DiceCoefficient()
        miou_metric = MeanIOU()
        
        labeled_iterator = iter(train_loader)
        for imgs_u in tqdm(unlabeled_loader, desc=f"Epoch {c_epochs}/{end_epochs}", unit="batch"):  
            start_time = time.time()
            with torch.no_grad():
                imgs_u = imgs_u.to(dev)
                model = model.eval()
                model.module.contrast=False
                
                p_masks = model(imgs_u)
                p_masks = p_masks.to('cpu').detach()
                imgs_u = imgs_u.to('cpu').detach() 
                
                #Since we use labeled data for PCGJCL as well
                try:
                    imgs_l, masks_l = next(labeled_iterator)
                except StopIteration:
                    labeled_iterator = iter(train_loader)
                    imgs_l, masks_l = next(labeled_iterator)
                
                #concatenating unlabeled and labeled sets
                c_masks = torch.cat([p_masks,masks_l],dim=0)
                c_imgs = torch.cat([imgs_u,imgs_l],dim=0)        
                
                #get classwise patch list
                patch_list = _get_patches(
                    imgs_u, p_masks,
                    classes=num_classes,
                    background=True,
                    img_size=img_size,
                    patch_size=patch_size
                )
                
                augmented_patch_list = batch_augment(patch_list, patch_size)
                aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
                qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]
            
            
            # labeled data
            model=model.train()
            teacher_model.train()
            model.module.contrast=False
            teacher_model.module.contrast = False   
            
            imgs_l, masks_l =imgs_l.to(dev), masks_l.to(dev)
            out_l = model(imgs_l)
            out_selected = out_l[:, 1:, :, :]  # shape: (16, C-1, 224, 224)
            masks_selected = masks_l[:, 1:, :, :]  # shape: (16, C-1, 224, 224)
            supervised_loss = criterion(out_selected, masks_selected)
            supervised_loss = supervised_loss.to(dev)
            dice_coeff.add_batch(out_selected, masks_selected)
            miou_metric.add_batch(out_selected, masks_selected)
            
            
            # Unlabeled data
            consistency_loss = consistency_regularization_CELoss(model, teacher_model, imgs_u)
            consistency_loss = consistency_loss.to(dev)
            
            
            model.module.contrast=True                
            student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, batch_size)
            
            teacher_model.module.contrast = True
            teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, True, batch_size)
            embd_queues.enqueue(teacher_embedding_list)
            
            PCGJCL_loss = simple_PCGJCL(student_emb_list, embd_queues, embedding_size, 0.2 , 4, psi=4096)
            PCGJCL_loss = PCGJCL_loss.to(dev)
            
            loss  = supervised_loss + contrastiveWeights * PCGJCL_loss + 4 * consistency_loss
            
            total_t_contrastive_loss += PCGJCL_loss.item()
            total_t_supervised_loss += supervised_loss.item()
            total_t_consistency_loss += consistency_loss.item()
            epoch_t_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            update_ema_variables(model, teacher_model, alpha=ema_alpha, global_step=c_epochs)
            end_time = time.time()

        avg_t_epoch_loss = epoch_t_loss / len(unlabeled_loader)
        avg_t_supervised_loss = total_t_supervised_loss / len(unlabeled_loader)
        avg_t_contrastive_loss = total_t_contrastive_loss / len(unlabeled_loader)
        avg_t_consistency_loss = total_t_consistency_loss / len(unlabeled_loader)
        
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
            t_consistency_loss = f"{avg_t_consistency_loss:.4f}",
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
