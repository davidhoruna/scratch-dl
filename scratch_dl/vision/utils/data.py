from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms
from scratch_dl.vision.configs.schemas import BaseConfig

import os


def load_data(cfg: BaseConfig):
    """
    Loads image data based on the specified folder structure.

    Args:
        cfg: Configuration object.
             `cfg.folder_structure` should be "ImageFolder" or "Flat".
             Other relevant attributes: `cfg.img_size`, `cfg.n_channels`.
        # You might also want to add args for transforms if you pass them directly
        # train_transforms (torchvision.transforms.Compose, optional): Transformations for training data.
        # val_transforms (torchvision.transforms.Compose, optional): Transformations for validation data.

    Returns:
        tuple: A tuple containing:
            - Dataset (torch.utils.data.Dataset): A PyTorch Dataset ready to be wrapped in a DataLoader.
            - Labels (dict): A dictionary mapping class names to integer indices (e.g., {'cat': 0, 'dog': 1}).

    Raises:
        ValueError: If `cfg.folder_structure` is not "ImageFolder" or "Flat".
    """
    cfg.data_dir=os.path.join(cfg.ROOT_DIR, "data", cfg.folder_name)

    structure=cfg.folder_structure

    # If the dataset follows the standard 'by_class' structure
    if structure.lower() == "imagefolder":
        return load_imagefolder(cfg)

    # For flat structure or other custom formats
    elif structure.lower() =="flat":
        return load_custom_dataset(cfg)

    else:
        raise ValueError("Unsupported structure type. Choose 'ImageFolder' or 'Flat'.")

def load_imagefolder(cfg: BaseConfig):
    dataset = ImageFolder(root=cfg.data_dir, transform=cfg.transforms)
    return dataset, dataset.class_to_idx

def load_custom_dataset(cfg:BaseConfig):
    # Example: Flat structure with labels in 'labels.txt'

    # Implement logic for a flat structure dataset
    # e.g., reading from a CSV or .txt for labels
    # Custom Dataset class can go here or implement any logic you need

    class CustomDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.image_paths = [p for ext in ['*.jpg', '*.png', '*.jpeg'] for p in data_dir.glob(ext)]  # Example: just looking for .jpg files
            self.labels = self.load_labels()

        def load_labels(self):
            # Assuming labels are in a text file with image paths and labels
            labels = {}
            with open(cfg.data_dir/ "labels.txt") as f:
                for line in f:
                    image_name, label = line.strip().split(',')
                    labels[image_name] = int(label)
            return labels

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            label = self.labels[image_path.name]
            transform = cfg.transforms
            return transform(image), label

        def __len__(self):
            return len(self.image_paths)

    dataset = CustomDataset(cfg.data_dir)
    return dataset, dataset.load_labels()
