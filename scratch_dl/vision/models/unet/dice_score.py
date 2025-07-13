import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Apply sigmoid to convert logits to probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors for calculation
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

