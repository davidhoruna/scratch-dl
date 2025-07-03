import datetime
import logging
import time
import argparse
from vision.utils import load_model, load_params, load_config, get_logger, load_data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch
import subprocess

logger = get_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent

available_models = ['unet', 'resnet', 'gans']
avlb_datasets = ['fruits', 'popular_street_foods']

parser = argparse.ArgumentParser("Train a vision model")
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-d', '--dataset', type=str, required=True)
args = parser.parse_args()


def train(model_name: str, dataset_dir: str = None):
    if model_name not in available_models:
        raise ValueError(f"Invalid model. Options: {', '.join(available_models)}")
    #if dataset_name not in avlb_datasets:
        raise ValueError(f"Invalid dataset. Options: {', '.join(avlb_datasets)}")

    dataset_dir = args.dataset
    cfg = load_config(model_name)

    if not dataset_dir:
        dataset_dir = cfg.data_dir
    
    train_dataset, test_dataset = load_data(dataset_dir, cfg.transform, cfg.split_ratio)
    
    model = load_model(model_name)
    params = load_params(model_name)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    criterion, optimizer_fn = params['loss'], params['optim']
    optimizer = optimizer_fn(model.parameters(), lr=cfg.learning_rate)

    log_dir = Path("runs") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir)
    subprocess.Popen(["tensorboard", "--logdir", str(log_dir)])

    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model.to(device)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss, correct, samples = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            samples += labels.size(0)

        train_acc = 100 * correct / samples
        writer.add_scalar("loss/train", total_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        logger.info(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Loss: {total_loss:.4f}")

        # Evaluation
        model.eval()
        correct, samples = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                samples += labels.size(0)

        test_acc = 100 * correct / samples
        writer.add_scalar("acc/test", test_acc, epoch)
        logger.info(f"[Epoch {epoch}] Test Acc: {test_acc:.2f}%")

    writer.close()


if __name__ == "__main__":
    train(args.model, args.dataset)
