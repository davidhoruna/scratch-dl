#!/usr/bin/env python3
"""
Test script to verify that the models can be instantiated correctly.
"""

import torch
from scratch_dl.vision.configs.schemas import UNetConfig, ResNetConfig
from scratch_dl.vision.models.unet.unet_model import UNet
from scratch_dl.vision.models.resnet.resnet_model import ResNet

def test_unet():
    print("Testing UNet...")
    config = UNetConfig()
    model = UNet(config)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"UNet input shape: {x.shape}")
    print(f"UNet output shape: {output.shape}")
    print("UNet test passed!\n")

def test_resnet():
    print("Testing ResNet...")
    config = ResNetConfig()
    model = ResNet(config)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"ResNet input shape: {x.shape}")
    print(f"ResNet output shape: {output.shape}")
    print("ResNet test passed!\n")

def test_config_update():
    print("Testing config update...")
    config = UNetConfig()
    print(f"Original lr: {config.lr}")
    
    # Simulate args
    class MockArgs:
        def __init__(self):
            self.lr = 1e-3
            self.epochs = 10
            self.batch_size = 16
    
    args = MockArgs()
    config.update_from_args(args)
    print(f"Updated lr: {config.lr}")
    print(f"Updated epochs: {config.epochs}")
    print("Config update test passed!\n")

if __name__ == "__main__":
    print("Running model tests...\n")
    test_config_update()
    test_unet()
    test_resnet()
    print("All tests passed!") 