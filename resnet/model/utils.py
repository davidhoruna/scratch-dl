import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # first conv layer 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # second conv layer        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # projection in case of channel or space mismatch

        self.shortcut = None

        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut != None:

            out += self.shortcut(x)
        else: 
            out += identity    
        out = self.relu(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.tranform = transform
        self.samples = []
        self.class_to_idx = {}
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.root_dir))

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((fpath, self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, img_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.tranform:
            image = self.tranform(image)
        return image, img_label

        
                