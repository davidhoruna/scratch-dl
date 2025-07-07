import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from vision.models.resnet.resnet_model import ResNet
from vision.configs import ResNetConfig
from PIL import Image
from pathlib import Path
from datetime import datetime
import logging

def load_data(data_dir, transform, split_ratio = 0.8):
    data_dir = Path(data_dir)
    transform = transform
    split_ratio = split_ratio

    if not data_dir.exists():
        raise Exception(f"Data directory not found: {data_dir}")

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if train_dir.exists() and test_dir.exists():
        train_subset = datasets.ImageFolder(train_dir, transform=transform)
        test_subset = datasets.ImageFolder(test_dir, transform=transform)
    else:
        dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
        split_dir = data_dir / "splits"
        split_dir.mkdir(exist_ok=True)

        train_idx_file = split_dir / "train_idx.pt"
        test_idx_file = split_dir / "test_idx.pt"

        if train_idx_file.exists() and test_idx_file.exists():
            train_indices = torch.load(train_idx_file)
            test_indices = torch.load(test_idx_file)
        else:
            torch.manual_seed(seed)
            indices = torch.randperm(len(dataset)).tolist()
            train_size = int(split_ratio * len(indices))
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            torch.save(train_indices, train_idx_file)
            torch.save(test_indices, test_idx_file)

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

    return train_subset, test_subset


def get_logger(name=__name__, level=logging.INFO):
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def load_model(model_name: str):
    if model_name == 'resnet':
        return ResNet()
    elif model_name == 'gan':
        raise NotImplementedError("GAN not implemented yet")
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_config(model_name: str):
    if model_name == 'resnet':
        return ResNetConfig()
    else:
        raise ValueError(f"No config for model: {model_name}")


def load_params(model_name: str):
    if model_name == 'resnet':
        return {
            'optim': Adam,
            'loss': CrossEntropyLoss()
        }
    else:
        raise ValueError(f"No params for model: {model_name}")
