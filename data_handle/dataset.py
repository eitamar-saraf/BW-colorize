import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip, RandomApply, Compose


class BWDataset(Dataset):
    def __init__(self, x_dir, y_dir, transforms=None):
        self.x = np.load(x_dir)
        self.y = np.load(y_dir)
        self.transforms = transforms
        self.augmentations = Compose([RandomApply(transforms=[RandomResizedCrop((224, 224))], p=0.25),
                                      RandomHorizontalFlip(0.5),
                                      RandomVerticalFlip(0.5)])

    def __getitem__(self, index):
        l_image = self.x[index]
        ab_image = self.y[index]

        l_image = np.expand_dims(l_image, 2)

        image = np.dstack((l_image, ab_image))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

        if self.augmentations:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()
            image = self.augmentations(image)
            image = image.numpy().transpose((1, 2, 0)).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_image = image[:, :, 0]
            ab_image = image[:, :, 1:]

        if self.transforms:
            l_image = self.transforms(l_image)
            ab_image = self.transforms(ab_image)

        return l_image, ab_image

    def __len__(self):
        return len(self.x)
