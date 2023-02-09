import cv2
import numpy as np

from Transform.Scalar import TorchStandardScaler
from Transform.normalize_inverse import NormalizeInverse


def reverse_transform(x):
    scaler = TorchStandardScaler()
    norm = NormalizeInverse(np.array([105.889, 131.027, 134.933]), np.array([71.90, 13.57, 19.448]))
    scaled_x = scaler.inverse_transform(x)
    scaled_x = norm(scaled_x)

    # transpose to (C, H, W) for cv2
    scaled_x = scaled_x.permute(1, 2, 0)

    # transform to numpy
    scaled_x = scaled_x.numpy()

    scaled_x = np.around(scaled_x, decimals=0).astype(np.uint8)
    scaled_x = cv2.cvtColor(scaled_x, cv2.COLOR_LAB2RGB)
    return scaled_x
