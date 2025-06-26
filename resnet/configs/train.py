from dataclasses import dataclass
from pathlib import Path
import argparse
import torch

@dataclass
class TrainingConfig:
    data_dir: Path = Path("data/fruits/fruits-360_100x100/fruits-360")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    num_workers: int = 2
    seed: int = 42
    save_checkpoint: bool = False
    save_freq: int = 5
    model_save_path: Path = Path("model.pth")

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=Path, default=cls.data_dir)
        parser.add_argument("--device", type=str, default=cls.device)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--learning_rate", type=float, default=cls.learning_rate)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--num_workers", type=int, default=cls.num_workers)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--save_checkpoint", action="store_true")
        parser.add_argument("--save_freq", type=int, default=cls.save_freq)
        parser.add_argument("--model_save_path", type=Path, default=cls.model_save_path)

        args = parser.parse_args()
        return cls(**vars(args))
