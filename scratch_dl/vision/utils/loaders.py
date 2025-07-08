import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from scratch_dl.vision.models.resnet.resnet_model import ResNet
from scratch_dl.vision.utils.data_loading import VisionDataset
from PIL import Image
from pathlib import Path
from datetime import datetime
from scratch_dl.vision.configs.schemas import BaseConfig, ResNetConfig, UNetConfig
import logging

def load_config(model_name:str):
    if model_name == "resnet":
        return ResNetConfig()
    if model_name == "unet":
        return UNetConfig()
def get_dataset(cfg: BaseConfig):
    kwargs = {
        "inputs_dir": cfg.inputs_dir,
        "labels_dir": cfg.labels_dir,
        "scale": getattr(cfg, "scale", 1.0)  # Default to 1.0 if not present
    }

    if cfg.model_name == "unet":
        return VisionDataset(task="segmentation", **kwargs)
    elif cfg.model_name == "resnet":
        return VisionDataset(task="classification", **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model}")


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



def load_optimloss(cfg: BaseConfig):
    optimloss = {'optim':None,'loss':None}
    model_name = cfg.model_name
    model = load_model(model_name)

    if cfg.model == 'resnet':
        return optim.Adam(model.parameters(), cfg.lr), CrossEntropyLoss()
    elif cfg.model == "unet":
        optimloss['optim'] = optim.RMSprop(model.parameters(),
                            lr=cfg.lr)
        optimloss['loss'] = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"No params for model: {cfg.model}")
    
    return optimloss