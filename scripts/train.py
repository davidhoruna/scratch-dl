# Copyright 2024 Your Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from resnet.configs.train import TrainingConfig
from resnet.model.utils import CustomDataset
from resnet.model.model import ResNet
from torch.utils.tensorboard import SummaryWriter
import datetime
import subprocess

from pathlib import Path
import sys
import logging


sys.path.append(str(Path(__file__).resolve().parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, filename=f"{ROOT_DIR}/logs")
logger = logging.getLogger("resnet-train")

def train(config: TrainingConfig):
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        # normalization for the 3 channels 
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    train_path = config.data_dir / "Training"
    test_path = config.data_dir / "Test"
    
    # we use our custom class
    train_dataset = CustomDataset(root=train_path, transform=transform)
    test_dataset = CustomDataset(root=test_path, transform=transform)


    n_classes = len(train_dataset.classes)

    # the iterator over the dataset
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    
    model = ResNet(n_classes=n_classes).to(device)
    # since there are several classes, we use CrossEntropy 
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # tensorboard setup
    log_dir = ROOT_DIR/ "runs"
    writer = SummaryWriter(log_dir/datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

    subprocess.Popen(["tensorboard", "--logdir", str(log_dir)])

    # start 
    for epoch in range(config.epochs):

        model.train()
        
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            # reset gradients 
            optimizer.zero_grad()
            # compute new gradients based on loss
            loss.backward()
            # update parameters
            optimizer.step()

            
            
            for name, param in model.named_parameters():
                # visualize parameters weights
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f"{name}.grad", param.grad, epoch)
                    writer.add_histogram(f"{name}.data", param.data, epoch)
            
            
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # total batch size
            total += labels.size(0)
            # find correct predicts
            correct += (predicted == labels).sum().item()
        # accuracy per epoch
        acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {total_loss:.4f} | Train Accuracy: {acc:.2f}%")

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        logger.info(f"Test Accuracy: {test_acc:.2f}%\n")

        if config.save_checkpoint and (epoch + 1) % config.save_freq == 0:
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Checkpoint saved at {config.model_save_path}")

        writer.add_scalar('Loss/train', total_loss,epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
    logger.info("Finished training. Epochs: %d | Final train accuracy: %.2f%%", config.epochs, acc)
    writer.close()
if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
