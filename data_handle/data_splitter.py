import os.path
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data() -> (np.ndarray, np.ndarray):
    l_channel = np.load('data/raw/l/gray_scale.npy')
    ab1 = np.load('data/raw/ab/ab1.npy')
    ab2 = np.load('data/raw/ab/ab2.npy')
    ab3 = np.load('data/raw/ab/ab3.npy')

    ab = np.concatenate((ab1, ab2, ab3), axis=0)

    return l_channel, ab


def split_data(l_channel: np.ndarray, ab: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    l_train, l_test, ab_train, ab_test = train_test_split(l_channel, ab, test_size=0.05, random_state=42)
    l_train, l_val, ab_train, ab_val = train_test_split(l_train, ab_train, test_size=0.05, random_state=42)
    return {'train': (l_train, ab_train), 'val': (l_val, ab_val), 'test': (l_test, ab_test)}


def save_data(data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    path = Path('data')
    for k, (x, y) in data.items():
        if not os.path.isdir(path.joinpath(k)):
            os.makedirs(path.joinpath(k))
        np.save(str(path.joinpath(k, 'x.npy')), x)
        np.save(str(path.joinpath(k, 'y.npy')), y)


def split_data_2_train_val_test():
    l_channel, ab = load_raw_data()
    data = split_data(l_channel, ab)
    save_data(data)
    print(f'Successfully split your data')
