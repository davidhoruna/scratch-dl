import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import wandb
import logging
import os

from scratch_dl.vision.utils.data import VisionDataset
from scratch_dl.vision.data_loaders.classification import ClassificationDataset

from scratch_dl.vision.utils.loaders import (
    load_config,
    load_model,
    load_optimloss,
    get_logger,
)
from scratch_dl.vision.configs.schemas import ResNetConfig, BaseConfig



def train_resnet(cfg: ResNetConfig):
    logger = get_logger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    
    dataset, labels = ClassificationDataset(cfg).load()  # Expecting (Dataset, label_dict)
    
    val_len = int(len(dataset) * cfg.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    
    cfg.n_classes = len(labels)
    
    model = load_model(cfg)
    model.to(device)

    optimizer, criterion = load_optimloss(cfg, model)

    wandb.init(project=cfg.project, config=vars(cfg), dir=cfg.ROOT_DIR)
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        correct_predictions_train = 0
        total_samples_train = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}") as pbar:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
                # Calculate training accuracy
                # For classification, assuming 'out' contains logits and 'y' contains class indices
                preds = torch.argmax(out, dim=1)
                correct_predictions_train += (preds == y).sum().item()
                total_samples_train += y.size(0)

                global_step += 1
                wandb.log({"train_loss": loss.item(), "step": global_step})
                wandb.log({"train_accuracy": correct_predictions_train / total_samples_train, "step": global_step})
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_predictions_train / total_samples_train
        wandb.log({"avg_train_loss_epoch": avg_train_loss, "train_accuracy_epoch": train_accuracy, "epoch": epoch})
        logger.info(f"Epoch {epoch + 1} training loss: {avg_train_loss:.4f}, training accuracy: {train_accuracy:.4f}")
        

        # Validation loop
        model.eval()
        val_loss = 0
        correct_predictions_val = 0
        total_samples_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                # Calculate validation accuracy
                preds = torch.argmax(out, dim=1)
                correct_predictions_val += (preds == y).sum().item()
                total_samples_val += y.size(0)
                wandb.log({"val_loss": loss.item()})


        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions_val / total_samples_val
        wandb.log({"avg_val_loss": avg_val_loss, "val_accuracy": val_accuracy, "epoch": epoch})
        logger.info(f"Validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}, validation accuracy: {val_accuracy:.4f}")

        if cfg.save_checkpoints:
            ckpt_path = Path(os.path.join(cfg.ROOT_DIR, 'checkpoints', cfg.checkpoint_dir))
            ckpt_path.mkdir(parents=True, exist_ok=True)
            if epoch == cfg.epochs:
                ckpt_file = ckpt_path / f"final_weights.pth"
            else:
                ckpt_file = ckpt_path / f"{cfg.model}_epoch{epoch+1}.pth"

            torch.save(model.state_dict(), ckpt_file)
            logger.info(f"Saved checkpoint: {ckpt_file}")
  