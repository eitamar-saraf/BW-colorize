import argparse
import torch

import numpy as np
import pytorch_lightning as pl

import cv2
from lightning_fabric import seed_everything

from data_handle.data_module import BWDataModule
from data_handle.data_splitter import split_data_2_train_val_test
from models.pix2pix.gan import GAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This app is used for BW-Colorize task')
    parser.add_argument('--action', type=str, choices=['split', 'train', 'eval'], default='train',
                        help='action for the app. train by default')
    args = parser.parse_args()

    if args.action == 'split':
        split_data_2_train_val_test()

    elif args.action == 'train':
        seed_everything(1235)
        torch.set_float32_matmul_precision('medium')
        train_path = 'data/train'
        val_path = 'data/val'
        data = BWDataModule(train_dir=train_path, val_dir=val_path)

        model = GAN()
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100)
        trainer.fit(model, data)

    elif args.action == 'eval':
        x = np.load('data/train/x.npy')
        y = np.load('data/train/y.npy')

        img = x[0]
        img = np.expand_dims(img, 2)
        ab = y[0]

        img = np.dstack((img, ab))
        rgb_sample = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        print(1)
