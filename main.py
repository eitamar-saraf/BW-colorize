import argparse

import numpy as np
import pytorch_lightning as pl

import cv2
import matplotlib.pyplot as plt

from data_handle.data_module import BWDataModule
from data_handle.data_splitter import split_data_2_train_val_test
from models.unet import Unet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This app is used for BW-Colorize task')
    parser.add_argument('--action', type=str, choices=['split', 'train', 'eval'], default='train',
                        help='action for the app. train by default')
    args = parser.parse_args()

    if args.action == 'split':
        split_data_2_train_val_test()

    elif args.action == 'train':
        train_path = 'dataset/train'
        val_path = 'dataset/val'
        data = BWDataModule(train_dir=train_path, val_dir=val_path)

        model = Unet(in_c=1, out_c=2)
        trainer = pl.Trainer(gpus=1)
        trainer.fit(model, data)

    elif args.action == 'eval':
        x = np.load('dataset/train/x.npy')
        y = np.load('dataset/train/y.npy')

        img = x[0]
        img = np.expand_dims(img, 2)
        ab = y[0]

        img = np.dstack((img, ab))
        rgb_sample = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        print(1)
