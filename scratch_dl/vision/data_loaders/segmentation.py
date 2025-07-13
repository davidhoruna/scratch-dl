import os
import logging
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from scratch_dl.vision.configs.schemas import UNetConfig
import numpy as np
import torch
logger = logging.getLogger(__name__)

class SegmentationDataset:
    def __init__(self, cfg: UNetConfig):
        self.cfg = cfg
        self.cfg.data_dir = os.path.join(cfg.ROOT_DIR, "data", cfg.folder_name)
        self.img_dir = os.path.join(self.cfg.data_dir, self.cfg.img_dir)
        self.mask_dir = os.path.join(self.cfg.data_dir, self.cfg.mask_dir)
        self.img_ext = ".jpg"
        self.mask_ext = ".png"

    def load(self):
        """
        Loads a segmentation dataset based on image and mask directories.

        Returns:
            tuple:
                - dataset: a PyTorch Dataset for segmentation.
                - labels: None (not applicable in segmentation use case).
        """
        logger.info(f"Data Directory: {self.cfg.data_dir}")
        logger.info(f"Image Directory: {self.img_dir}")
        logger.info(f"Mask Directory: {self.mask_dir}")

        class SegDataset(Dataset):
            def __init__(self, img_dir, mask_dir, transform_img=None, transform_mask=None, img_ext=".jpg", mask_ext=".png"):
                self.img_dir = Path(img_dir)
                self.mask_dir = Path(mask_dir)
                self.img_ext = img_ext
                self.mask_ext = mask_ext
                self.image_paths = sorted([p for p in self.img_dir.glob(f"*{img_ext}")])
                self.transform_img = transform_img
                self.transform_mask = transform_mask

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                img_stem = img_path.stem
                mask_path = self.mask_dir / f"{img_stem}{self.mask_ext}"

                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")  # grayscale

                mask = np.array(mask).astype(np.uint8)
                mask = torch.from_numpy(mask).long()  # [H, W] class indices

                if self.transform_img:
                    image = self.transform_img(image)
                if self.transform_mask:
                    mask = self.transform_mask(mask)

                return image, mask


        dataset = SegDataset(
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            transform_img=self.cfg.transform_img,
            transform_mask=self.cfg.transform_mask,
            img_ext=self.img_ext,
            mask_ext=self.mask_ext
        )

        return dataset
