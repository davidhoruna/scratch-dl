from dataclasses import dataclass, field
import os
from torchvision import transforms
from typing import Optional, Callable, Tuple
from definitons import ROOT_DIR  # Ensure this is defined globally before use
import yaml


@dataclass
class BaseConfig:
    folder_structure: str = ""
    folder_name: str = ""
    ROOT_DIR: str = ROOT_DIR
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-5
    val_split: float = 0.1
    val_percent: float = 0.1
    save_checkpoints: bool = True
    project: str = "vision-training"
    checkpoint_dir: str = "vision-training"
    img_scale: float = 0.5
    img_size: Tuple[int, int] = (224, 224)
    amp: bool = False
    weight_decay: float = 1e-4
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    model: str = "unet"
    model_name: str = "unet"
    n_classes: int = 2
    n_channels: int = 3
    bilinear: bool = False

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, name: str, path: str):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, f"{name}_config.yaml")
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


@dataclass
class UNetConfig(BaseConfig):
    task: str = "segmentation"
    img_dir: str = "Image"
    mask_dir: str = "Mask"
    transform_img: Optional[Callable] = field(default_factory=lambda: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    transform_mask: Optional[Callable] = field(default_factory=lambda: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))


@dataclass
class ResNetConfig(BaseConfig):
    task: str = "classification"
    n_classes: int = 151
    num_blocks: int = 5
    transform_img: Optional[Callable] = field(default_factory=lambda: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
