from dataclasses import dataclass
import argparse
import os
import yaml

# Base config class for common parameters
@dataclass
class BaseConfig:
    inputs_dir = "data/images"
    labels_dir = "data/labels"
    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-5
    val_percent: float = 0.1
    save_checkpoint: bool = True
    img_scale: float = 0.5
    amp: bool = False
    weight_decay: float = 1e-8
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    model_name: str = "unet"  # Default to UNet model
    n_classes: int = 2  # Default number of output classes
    n_channels: int = 3  # Default input channels
    bilinear: bool = False  # Bilinear upsampling option
    
    def save_config(self, name: str, path: str):
        """Save configuration to yaml file"""
        # Create config directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, f"{name}_config.yaml")
        
        # Convert dataclass to dict
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # Save to yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Config for UNet model
@dataclass
class UNetConfig(BaseConfig):
    pass  # Inherit everything from BaseConfig

# Config for ResNet model
@dataclass
class ResNetConfig(BaseConfig):
    model_name: str = "resnet"
    # Add ResNet specific config fields here if needed
    num_blocks: int = 5
