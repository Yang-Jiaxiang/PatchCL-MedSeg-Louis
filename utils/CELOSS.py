import segmentation_models_pytorch as smp
import torch

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CE_loss(smp.utils.losses.CrossEntropyLoss):
    def __init__(self, num_classes, image_size):
        super(CE_loss,self).__init__()
        super(CE_loss, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.ce_loss = smp.utils.losses.CrossEntropyLoss()
        self.matrix_mult = torch.ones(num_classes, image_size, image_size) * torch.arange(num_classes).view(num_classes, 1, 1)

    def forward(self,prediction,target):
        self.matrix_mult=(self.matrix_mult).to(dev)
        target = target*self.matrix_mult.unsqueeze(0)
        target = (torch.sum(target,dim=1)).type(torch.long)
        return self.ce_loss(prediction,target)
