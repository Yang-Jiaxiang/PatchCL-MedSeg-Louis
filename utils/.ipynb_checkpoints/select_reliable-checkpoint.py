import torch 

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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