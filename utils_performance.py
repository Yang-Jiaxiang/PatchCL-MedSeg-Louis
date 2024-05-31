import torch

class DiceCoefficient:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.dice_scores = []

    def add_batch(self, predictions, gts):
        # Apply threshold
        predictions = (predictions > self.threshold).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * gts).sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (predictions.sum(dim=(2, 3)) + gts.sum(dim=(2, 3)) + 1e-6)
        self.dice_scores.append(dice.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.dice_scores)).item()


class Accuracy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def add_batch(self, predictions, gts):
        # Apply threshold
        predictions = (predictions > self.threshold).float()
        
        # Calculate accuracy
        correct = (predictions == gts).float()
        self.correct += correct.sum().item()
        self.total += correct.numel()

    def evaluate(self):
        return self.correct / self.total


class MeanIOU:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.miou_scores = []

    def add_batch(self, predictions, gts):
        # Apply threshold
        predictions = (predictions > self.threshold).float()
        
        # Calculate intersection and union
        intersection = (predictions * gts).sum(dim=(2, 3))
        union = (predictions + gts).sum(dim=(2, 3)) - intersection
        
        # Calculate mIOU
        miou = (intersection + 1e-6) / (union + 1e-6)
        self.miou_scores.append(miou.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.miou_scores)).item()
