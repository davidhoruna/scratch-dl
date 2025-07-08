import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import wandb
import logging
import os

from scratch_dl.vision.utils.loaders import (
    load_config,
    load_model,
    load_optimloss,
    get_dataset,
    get_logger,
)

def train(cfg):
    logger = get_logger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = get_dataset(cfg)
    val_len = int(len(dataset) * cfg.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    model = load_model(cfg.model)
    model.to(device)

    optimizer, criterion = load_optimloss(cfg)

    
    scaler = GradScaler(device, enabled=cfg.amp)

    wandb.init(project=cfg.project, config=vars(cfg))
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}") as pbar:
            for batch, (x,y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=cfg.amp):
                    out = model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                global_step += 1
                wandb.log({"train_loss": loss.item(), "step": global_step})
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        # Optionally: add validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (x,y) in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=cfg.amp):
                    out = model(x)
                    val_loss += criterion(out, y).item()

        val_loss /= len(val_loader)
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        logger.info(f"Validation loss after epoch {epoch + 1}: {val_loss}")

        if cfg.save_checkpoints:
            ckpt_path = Path(cfg.checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path / f"{cfg.model}_epoch{epoch+1}.pth")
            logger.info(f"Saved checkpoint: {ckpt_path / f'{cfg.model}_epoch{epoch+1}.pth'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='unet', type=str, required=False, help="Model name: unet, resnet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--project", type=str, default="vision-training")
    parser.add_argument("--images_dir", type=str, default="./data/images")
    parser.add_argument("--mask_dir", type=str, default="./data/masks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.model)

    # Override config values from CLI
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    train(cfg)
