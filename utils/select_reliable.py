import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset


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

            teacher_labels = teacher_outputs.argmax(dim=1)
            consistency_loss = criterion(student_outputs, teacher_labels, reduction='none')

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

