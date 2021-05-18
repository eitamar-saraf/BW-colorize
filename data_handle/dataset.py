from torch.utils.data import Dataset
import numpy as np
import cv2 as cv


class BWDataset(Dataset):
    def __init__(self, x_dir, y_dir, transforms=None):
        self.x = np.load(x_dir)
        self.y = np.load(y_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        l_image = self.x[index]
        ab_image = self.y[index]

        l_image = cv.resize(l_image, (192, 192), interpolation=cv.INTER_AREA)
        ab_image = cv.resize(ab_image, (192, 192), interpolation=cv.INTER_AREA)

        if self.transforms:
            l_image = self.transforms(l_image)
            ab_image = self.transforms(ab_image)

        return l_image, ab_image

    def __len__(self):
        return len(self.x)
