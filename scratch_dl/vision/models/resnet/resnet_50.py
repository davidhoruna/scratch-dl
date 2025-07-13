import torch.nn as nn
from scratch_dl.vision.models.resnet.resnet_blocks import ResidualBlock, Bottleneck
from scratch_dl.vision.configs.schemas import ResNetConfig, BaseConfig


class ResNet50(nn.Module):
    def __init__(self, cfg: BaseConfig = None):
        super(ResNet50, self).__init__()
        if cfg == None:
            raise ValueError("Provide config file")
        
        self.n_classes = cfg.n_classes
        

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # we create a stack of residual layers
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = Bottleneck
        layers_config = [3,4,6,3]
        self.layer0=self._make_layer(Bottleneck, 64, layers_config[0], stride=1)
        self.layer1=self._make_layer(Bottleneck, 128, layers_config[1], stride=2)
        self.layer2=self._make_layer(Bottleneck, 256, layers_config[2], stride=2)
        self.layer3=self._make_layer(Bottleneck, 512, layers_config[3], stride=2)
        in_features = 512*block.expansion
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        # final layer for preds
        self.fc = nn.Linear(in_features, self.n_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # this will create a sequential block of layers 
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        Creates a sequential container composed of stacked residual blocks
        Args:
            block (nn.Module): Residual Block class
            out_channels (int): Number of output channels per each block
            num_blocks (int): Number of blocks to stack in the layer
            stride (int): Stride of the first block. 
        Returns:
            nn.Sequential: A sequential container of residual blocks
        """
        
        downsample=None
        expansion = block.expansion if hasattr(block, 'expansion') else 1
        if stride != 1 or self.in_channels != out_channels*expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * expansion # Update in_channels for next block
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels)) # Subsequent blocks in a layer typically have stride 1 and no downsample

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        
        # pass through residual layers
        out = self.layer0(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)

        # global avg pooling 
        out=self.avgpool(out)
        
        # flattens the output to match the linear layer
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