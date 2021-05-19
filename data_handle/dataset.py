import torch
from torch.utils.data import Dataset
import numpy as np


class BWDataset(Dataset):
    def __init__(self, x_dir, y_dir, transforms=None):
        self.x = np.load(x_dir)
        self.y = np.load(y_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        l_image = self.x[index]
        ab_image = self.y[index]

        l_image = np.expand_dims(l_image, 2)

        if self.transforms:
            l_image = self.transforms(l_image)

        return l_image, torch.Tensor(ab_image)

    def __len__(self):
        return len(self.x)
