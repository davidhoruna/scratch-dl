# root/train.py
from scratch_dl.vision.models.resnet.train import train_resnet
from scratch_dl.vision.models.unet.train import train_unet


from scratch_dl.vision.configs.schemas import BaseConfig, ResNetConfig, UNetConfig
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default='resnet50', type=str, help="Model name: unet, resnet")
    parser.add_argument("--folder_name", default='PokemonData', type=str, help="Folder name in data/ dir")
    parser.add_argument("--folder_structure", default='ImageFolder', type=str, help="Folder structure (by_class or flat)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--project", type=str, default="pokemon-classification")
    return parser.parse_args()

args = parse_args()

# Dispatch based on model type
if args.model_name == "unet":
    cfg = UNetConfig()
    cfg.update_from_args(args)
    train_unet(cfg)
elif args.model_name == "resnet50":

    cfg = ResNetConfig()
    cfg.update_from_args(args)
    train_resnet(cfg)


elif args.model == "ppo":
    """ 
    cfg = PPOConfig().up
    train_rl(cfg) """


