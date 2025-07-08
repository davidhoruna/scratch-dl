from vision.configs.schemas import BaseConfig
from vision.models.resnet.resnet_model import ResNet
from vision.models.unet.unet_model import UNet


def load_model(cfg:BaseConfig):
    model_name = cfg.model.lower()
    if model_name == "resnet":
        return ResNet(cfg)
    if model_name == "unet":
        return UNet(cfg)