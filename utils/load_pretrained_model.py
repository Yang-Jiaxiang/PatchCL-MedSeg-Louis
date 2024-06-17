import torch

def load_pretrained_model(model, teacher_model, save_model_path, epoch):
    model_path = f"{save_model_path}{epoch}-s.pth"
    teacher_model_path = f"{save_model_path}{epoch}-t.pth"
    print('model_path: ', model_path)
    print('teacher_model_path: ', teacher_model_path)
    print("")

    model = torch.load(model_path)
    teacher_model = torch.load(teacher_model_path)

    model.eval()
    teacher_model.eval()
    return model, teacher_model