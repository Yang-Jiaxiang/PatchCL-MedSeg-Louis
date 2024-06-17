import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
from utils.DISLOSS import DiceLoss
import torch.nn.functional as F
import torchvision.transforms as transforms


def softmax_mse_loss(student_outputs, teacher_outputs):
    student_softmax = torch.nn.functional.softmax(student_outputs, dim=1)
    teacher_softmax = torch.nn.functional.softmax(teacher_outputs, dim=1)
    return torch.mean((student_softmax - teacher_softmax) ** 2, dim=1)


def consistency_regularization_CELoss(model, teacher_model, imgs):
    # Define strong and weak augmentations
    strong_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.5)
        # Add more morphological transformations if needed
    ])

    weak_aug = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 30)),  # Random rotation within 0-30 degrees
        transforms.RandomCrop(size=(imgs.shape[2], imgs.shape[3]))  # Random crop, set size as needed
        # Add more augmentations if needed
    ])

    # Apply augmentations to the images
    weak_aug_imgs = weak_aug(imgs)
    strong_aug_imgs = strong_aug(imgs)
    
    # Get predictions from the model and teacher model
    model_preds = model(strong_aug_imgs)
    teacher_preds = teacher_model(weak_aug_imgs)
    
    # Use softmax on both predictions to get probability distributions
    model_probs = F.softmax(model_preds, dim=1)
    teacher_probs = F.softmax(teacher_preds, dim=1)
    
    # Flatten the tensors to match the input requirements of F.cross_entropy
    model_probs = model_probs.permute(0, 2, 3, 1).reshape(-1, model_probs.shape[1])
    teacher_probs = teacher_probs.permute(0, 2, 3, 1).reshape(-1, teacher_probs.shape[1])

    # Compute cross-entropy loss
    consistency_loss = torch.mean(torch.sum(-teacher_probs * torch.log(model_probs + 1e-10), dim=-1))

    return consistency_loss

def select_reliable(model, teacher_model, data_loader, num_classes, threshold=0.1, device='cuda'):
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
            noise = torch.clamp(torch.randn_like(imgs) * 0.05, -0.1, 0.1)  # 調整噪聲大小
            ema_inputs = imgs + noise

            student_outputs = model(imgs)
            teacher_outputs = teacher_model(ema_inputs)
            
            # 計算基於 softmax 的 MSE 損失
            consistency_loss = softmax_mse_loss(student_outputs, teacher_outputs)
            
            student_outputs_one_hot = torch.nn.functional.one_hot(student_outputs.argmax(dim=1), num_classes=num_classes)
            student_outputs_one_hot = student_outputs_one_hot.permute(0, 3, 1, 2).float()

            for img, output, loss in zip(imgs, student_outputs_one_hot, consistency_loss):
                if loss.mean().item() < threshold:
                    reliable_images.append(img.cpu().numpy())
                    reliable_outputs.append(output.cpu().numpy())
                else:
                    remaining_images.append(img.cpu().numpy())

    reliable_images_tensor = torch.tensor(reliable_images)
    reliable_outputs_tensor = torch.tensor(reliable_outputs)
    reliable_dataset = TensorDataset(reliable_images_tensor, reliable_outputs_tensor)
    remaining_dataset = TensorDataset(torch.tensor(remaining_images))
    return reliable_dataset, remaining_dataset


def Label(model, data_loader, num_classes, device='cuda'):
    tbar = tqdm(data_loader)
    images = []
    outputs = []
    
    for batch in tbar:
        print(f"Batch type: {type(batch)}")  # 打印批次的類型
        if isinstance(batch, list):
            print(f"Batch contains {len(batch)} elements")
            imgs = batch[0]  # 假設圖像數據在第一個位置
        elif isinstance(batch, torch.Tensor):
            imgs = batch
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        
        print(f"Images type: {type(imgs)}, shape: {imgs.shape}")  # 打印圖像數據的類型和形狀
        
        imgs = imgs.to(device)
        output = model(imgs)
        
        outputs_one_hot = torch.nn.functional.one_hot(output.argmax(dim=1), num_classes=num_classes)
        outputs_one_hot = outputs_one_hot.permute(0, 3, 1, 2).float()
        
        for img, output in zip(imgs, outputs_one_hot):
            images.append(img.cpu().numpy())
            outputs.append(output.cpu().numpy())
            
    images_tensor = torch.tensor(images)
    outputs_tensor = torch.tensor(outputs)
    
    return TensorDataset(images_tensor, outputs_tensor)

