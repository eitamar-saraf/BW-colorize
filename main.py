import argparse

from torch.utils.data import DataLoader

from data_handle.data_splitter import split_data_2_train_val_test
from data_handle.dataset import BWDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This app is used for BW-Colorize task')
    parser.add_argument('--action', type=str, choices=['split', 'train', 'eval'], default='train',
                        help='action for the app. train by default')
    args = parser.parse_args()

    if args.action == 'split':
        split_data_2_train_val_test()

    batch_size = 64
    train_dataset = BWDataset(x_dir='dataset/l/gray_scale.npy', y_dir='dataset/ab/ab1.npy')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
