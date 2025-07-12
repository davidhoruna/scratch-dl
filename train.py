# root/train.py
from scratch_dl.vision.train_vision import train as train_vision

from scratch_dl.vision.configs.schemas import BaseConfig, ResNetConfig, UNetConfig
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='resnet', type=str, help="Model name: unet, resnet")
    parser.add_argument("--folder_name", default='PokemonData', type=str, help="Folder name in data/ dir")
    parser.add_argument("--folder_structure", default='by_class', type=str, help="Folder structure (by_class or flat)")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--project", type=str, default="pokemon-classification")
    parser.add_argument("--images_dir", type=str, default="./data/images")
    parser.add_argument("--mask_dir", type=str, default="./data/masks")
    return parser.parse_args()

args = parse_args()

# Dispatch based on model type
if args.model == "unet":
    cfg = UNetConfig()
    cfg.update_from_args(args)
    train_vision(cfg)
elif args.model == "resnet":

    cfg = ResNetConfig()
    cfg.update_from_args(args)
    train_vision(cfg)


elif args.model == "ppo":
    """ 
    cfg = PPOConfig().up
    train_rl(cfg) """


