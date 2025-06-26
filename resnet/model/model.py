import torch.nn as nn
from .utils import ResidualBlock

class ResNet(nn.Module):
    def __init__(self, n_classes = 10):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0=self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer1=self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer2=self._make_layer(ResidualBlock, 256, 2)
        self.layer3=self._make_layer(ResidualBlock, 512, 2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, n_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        
        strides = [stride] + [1]*(num_blocks-1)


        layers=[]
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        
        out = self.layer0(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avgpool(out)
        
        # 
        out=out.view(out.size(0),-1)
        out=self.fc(out)

        

        return out


    """ Input        → [3, 256, 256]
Conv1        → [64, 128, 128]
MaxPool      → [64, 64, 64]
Layer0       → [64, 64, 64]
Layer1       → [128, 32, 32]
Layer2       → [256, 32, 32]
Layer3       → [512, 32, 32]
AvgPool      → [512, 1, 1]
Flatten+FC   → [512] → [n_classes] """