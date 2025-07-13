import os
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from scratch_dl.vision.configs.schemas import BaseConfig

logger = logging.getLogger(__name__)

class ClassificationDataset:
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.cfg.data_dir = os.path.join(cfg.ROOT_DIR, "data", cfg.folder_name)
        self.structure = cfg.folder_structure.lower()
        logger.info(f"Data Directory: {self.cfg.data_dir}")

    def load(self):
        """Load the dataset based on structure."""
        if self.structure == "imagefolder":
            return self._load_imagefolder()
        elif self.structure == "flat":
            return self._load_flat()
        else:
            raise ValueError("Unsupported structure type. Use 'ImageFolder' or 'Flat'.")

    def _load_imagefolder(self):
        dataset = ImageFolder(root=self.cfg.data_dir, transform=self.cfg.transforms)
        return dataset, dataset.class_to_idx

    def _load_flat(self):
        class CustomFlatDataset(Dataset):
            def __init__(self, data_dir, transform):
                self.data_dir = Path(data_dir)
                self.transform = transform
                self.image_paths = sorted([
                    p for ext in ["*.jpg", "*.jpeg", "*.png"]
                    for p in self.data_dir.glob(ext)
                ])
                self.labels = self._load_labels()

            def _load_labels(self):
                labels_path = self.data_dir / "labels.txt"
                if not labels_path.exists():
                    raise FileNotFoundError(f"Missing labels.txt in {self.data_dir}")
                labels = {}
                with open(labels_path, "r") as f:
                    for line in f:
                        name, label = line.strip().split(',')
                        labels[name] = int(label)
                return labels

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert("RGB")
                label = self.labels.get(image_path.name)
                if label is None:
                    raise KeyError(f"Label for {image_path.name} not found in labels.txt")
                return self.transform(image), label

        dataset = CustomFlatDataset(self.cfg.data_dir, self.cfg.transforms)
        return dataset, dataset.labels
