import cv2
import numpy as np

from data_handle.Scalar import TorchStandardScaler


def reverse_transform(x):
    scaler = TorchStandardScaler()
    scaled_x = scaler.inverse_transform(x)
    scaled_x = scaled_x * 255
    scaled_x = np.around(scaled_x, decimals=0).astype(np.uint8)
    scaled_x = cv2.cvtColor(scaled_x, cv2.COLOR_LAB2RGB)
    return scaled_x
