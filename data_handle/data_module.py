from os import path
from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import transforms

import numpy as np

from Transform.Scalar import TorchStandardScaler
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
        self.mean = np.array([105.889, 131.027, 134.933])
        self.std = np.array([71.90, 13.57, 19.448])

    def prepare_data(self):
        self.trans = transforms.Compose(
            [transforms.Normalize(self.mean, self.std),
             TorchStandardScaler()])

        # self.trans = transforms.Compose([transforms.ToTensor(), TorchStandardScaler()])

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = BWDataset(x_dir=path.join(self.train_dir, 'x.npy'),
                                       y_dir=path.join(self.train_dir, 'y.npy'),
                                       transforms=self.trans)
        self.val_dataset = BWDataset(x_dir=path.join(self.val_dir, 'x.npy'),
                                     y_dir=path.join(self.val_dir, 'y.npy'),
                                     transforms=self.trans)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
