import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join, splitext


def load_image(path):
    ext = splitext(path)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(path))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(path).numpy())
    else:
        return Image.open(path)


class VisionDataset(Dataset):
    def __init__(
        self,
        inputs_dir,
        labels_dir=None,
        task="classification",  # or "segmentation"
        scale: float = 1.0,
        transform=None,
        label_suffix="",
        input_suffix="",
        preprocess_fn=None
    ):
        self.inputs_dir = Path(inputs_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.scale = scale
        self.transform = transform
        self.task = task
        self.label_suffix = label_suffix
        self.input_suffix = input_suffix
        self.preprocess_fn = preprocess_fn

        self.ids = [
            splitext(f)[0]
            for f in listdir(inputs_dir)
            if isfile(join(inputs_dir, f)) and not f.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(f"No input files found in {inputs_dir}")

        logging.info(f"Loaded {len(self.ids)} samples from {inputs_dir}")

    def __len__(self):
        return len(self.ids)

    def resize(self, img: Image.Image) -> Image.Image:
        if self.scale != 1.0:
            w, h = img.size
            newW, newH = int(w * self.scale), int(h * self.scale)
            return img.resize((newW, newH), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Load input image
        input_path = list(self.inputs_dir.glob(name + self.input_suffix + ".*"))[0]
        image = load_image(input_path)
        image = self.resize(image)

        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        else:
            image = np.asarray(image)
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            else:
                image = image.transpose((2, 0, 1))  # HWC -> CHW
            image = torch.as_tensor(image.copy()).float() / 255.0

        out = {"image": image.contiguous()}

        # Optional: load labels (segmentation masks or class labels)
        if self.labels_dir:
            label_path = list(self.labels_dir.glob(name + self.label_suffix + ".*"))[0]
            label = load_image(label_path)
            label = self.resize(label)

            if self.task == "segmentation":
                label = np.asarray(label)
                if label.ndim == 3:
                    label = label[..., 0]  # Assume mask is grayscale in RGB
                label = torch.as_tensor(label.copy()).long()
                out["mask"] = label.contiguous()

            elif self.task == "classification":
                label = int(np.asarray(label))  # e.g., label stored as scalar image
                out["label"] = torch.tensor(label).long()

        return out
