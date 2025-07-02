from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    data_dir: Path = Path("data/")
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 50
    num_workers: int = 4
    save_model_path: Path = Path("checkpoints/unet.pth")
    log_dir: Path = Path("runs/unet/")
    device: str = "cuda"  # or "cpu"
    seed: int = 42
