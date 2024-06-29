import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class AestheticDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        label = self.labels.iloc[idx, 1]

        try:
            image = Image.open(img_name).convert('RGB')
        except (OSError, IOError):
            print(f"Error loading image {img_name}. Skipping this image.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
