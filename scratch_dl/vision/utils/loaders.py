import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from scratch_dl.vision.models.unet.unet_model import UNet
from PIL import Image
from pathlib import Path
from datetime import datetime
from scratch_dl.vision.configs.schemas import BaseConfig, ResNetConfig, UNetConfig
import logging

def load_config(model_name:str, n_classes: int = 0, ):
    if model_name == "resnet":
        return ResNetConfig()
    elif model_name == "unet":
        return UNetConfig()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


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


def load_model(cfg: BaseConfig = None):
    model_name = cfg.model_name
    if model_name == 'resnet':
        return ResNet(cfg)
    elif model_name == 'unet':
        return UNet(cfg)
    elif model_name == 'gan':
        raise NotImplementedError("GAN not implemented yet")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

from torch import nn
def load_optimloss(cfg: BaseConfig, model: nn.Module):
    model_name = cfg.model.lower()
    params = model.parameters()
    if model_name == 'resnet':
        # Return optimizer class and criterion
        optimizer_class = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = CrossEntropyLoss()
        return optimizer_class, criterion
    elif model_name == "unet":
        # Return optimizer class and criterion
        optimizer_class = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)  # Changed from RMSprop to Adam for consistency
        if cfg.n_classes > 1:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
        return optimizer_class, criterion
    else:
        raise ValueError(f"No params for model: {cfg.model}")