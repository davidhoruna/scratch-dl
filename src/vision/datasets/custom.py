from torch.utils.data import Dataset
import os
from PIL import Image 
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.tranform = transform
        self.samples = []
        self.class_to_idx = {}
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.root_dir))

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((fpath, self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, img_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.tranform:
            image = self.tranform(image)
        return image, img_label

        
        