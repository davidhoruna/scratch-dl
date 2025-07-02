import torch.nn as nn
from .model.blocks import DownBlock, UpBlock, DoubleConv

class UNet(nn.Module):
    """
    Modular U-Net implementation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(DownBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        rev_features = list(reversed(features))
        for feature in rev_features:
            self.decoder.append(UpBlock(prev_channels, feature))
            prev_channels = feature

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            skip, x = down(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx, up in enumerate(self.decoder):
            x = up(x, skip_connections[idx])

        return self.final_conv(x)
