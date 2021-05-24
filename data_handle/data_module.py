from os import path
from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import transforms

import numpy as np

from data_handle.dataset import BWDataset


class BWDataModule(pl.LightningDataModule):

    def __init__(self, train_dir: str, val_dir: str, batch_size: int = 64):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.trans = None

    def prepare_data(self):
        x = np.load(path.join(self.train_dir, 'x.npy'))
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[x.mean() / 255], std=[x.std() / 255])])

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = BWDataset(x_dir=path.join(self.train_dir, 'x.npy'),
                                       y_dir=path.join(self.train_dir, 'y.npy'),
                                       transforms=self.trans)
        self.val_dataset = BWDataset(x_dir=path.join(self.val_dir, 'x.npy'),
                                     y_dir=path.join(self.val_dir, 'y.npy'),
                                     transforms=self.trans)

    def transfer_batch_to_device(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)
