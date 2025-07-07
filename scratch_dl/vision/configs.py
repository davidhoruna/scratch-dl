from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms
@dataclass
class UnetConfig:
    data_dir: Path = Path("data/fruits/fruits-360_100x100/fruits-360")
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 50
    num_workers: int = 4
    seed: int = 42
    split_ratio: float = 0.8
    device: str = "cuda"
    save_model_path: Path = Path("checkpoints/unet.pth")
    log_dir: Path = Path("runs/unet/")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
@dataclass
class ResNetConfig:
    data_dir: Path = Path("data/fruits/fruits-360_100x100/fruits-360")
    image_size: tuple = (100, 100)
    num_classes: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    split_ratio: float = 0.8
    device: str = "cuda"
    save_model_path: Path = Path("checkpoints/resnet.pth")
    log_dir: Path = Path("runs/resnet/")

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
