
import os
from scratch_dl.vision.data_loaders.segmentation import SegmentationDataset
from scratch_dl.vision.configs.schemas import BaseConfig
import logging

logger = logging.getLogger(__name__)

class VisionDataset:
    
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

        return self.load_data(cfg)
    def load_data(self):
        """
        Loads image data based on the specified task and folder structure.
        Args:
            cfg: Configuration object.
                For classification `cfg.folder_structure` should be "ImageFolder" or "Flat".
        """
        
        if self.cfg.task == "classification": 
            return ClassificationDataset(self.cfg)
        elif self.cfg.task == "segmentation": 
            return  SegmentationDataset(self.cfg)




