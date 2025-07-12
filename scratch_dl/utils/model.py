from scratch_dl.vision.configs.schemas import BaseConfig
from scratch_dl.vision.models.resnet.resnet_model import ResNet
from scratch_dl.vision.models.unet.unet_model import UNet


def load_model(cfg: BaseConfig):
    model_name = cfg.model.lower()
    if model_name == "resnet":
        return ResNet(cfg)
    elif model_name == "unet":
        return UNet(cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")