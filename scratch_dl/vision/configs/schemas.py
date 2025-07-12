from dataclasses import dataclass
import os
# Assuming ROOT_DIR is correctly defined and imported from definitons
from definitons import ROOT_DIR 
import yaml
from torchvision import transforms

# Base config class for common parameters
@dataclass
class BaseConfig:
    folder_structure: str = ""
    folder_name: str = ""
    ROOT_DIR = ROOT_DIR # Class attribute, assuming it's loaded correctly
    epochs: int = 50 # Increased default for more realistic training
    batch_size: int = 32 # Significantly increased for efficiency and stability
    lr: float = 1e-5 # Lowered for more stable training
    # Removed 'lr' alias here to avoid ambiguity
    val_split: float = 0.1
    val_percent: float = 0.1  # Alias for val_split (keep if desired, but clarify usage)
    save_checkpoints: bool = True  # Alias for save_checkpoint (keep if desired)
    project: str = "vision-training"
    checkpoint_dir: str = project
    img_scale: float = 0.5
    img_size: tuple = (224, 224)  # Changed to standard ImageNet size
    amp: bool = False
    
    weight_decay: float = 1e-4  # Increased for better regularization
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    model: str = "unet"  # Model name
    model_name: str = "unet"  # Alias for model (keep if desired)
    n_classes: int = 2  # Default number of output classes
    n_channels: int = 3  # Default input channels
    bilinear: bool = False  # Bilinear upsampling option
    transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    def update_from_args(self, args):
        """Update config from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        
    
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

# Config for UNet model (no changes needed beyond BaseConfig updates)
@dataclass
class UNetConfig(BaseConfig):
    model: str = "unet"
    model_name: str = "unet"
    n_classes: int = 2
    n_channels: int = 3
    bilinear: bool = False

# Config for ResNet model (no changes needed beyond BaseConfig updates)
@dataclass
class ResNetConfig(BaseConfig):
    model: str = "resnet"
    model_name: str = "resnet"
    n_classes: int = 151  # Updated for Pokemon dataset (151 Pokemon classes)
    n_channels: int = 3
    num_blocks: int = 5