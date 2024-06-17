import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_bn_stats(model, train_loader):
    model.train()
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(dev)
            _ = model(imgs)