import numpy as np
import torch


class TorchStandardScaler:

    def __init__(self):
        self.b = torch.as_tensor([1, 1, 1], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        self.a = torch.as_tensor([-1, -1, -1], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        self.min = torch.as_tensor([-1.4727, -9.6556, -6.93814], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        self.max = torch.as_tensor([2.0738, 9.1358, 6.9737], dtype=torch.float32).unsqueeze(1).unsqueeze(2)

    def __call__(self, x):
        scaled_x = self.a + (x - self.min) * (self.b - self.a) / (self.max - self.min)
        return scaled_x

    def inverse_transform(self, x):
        unscaled_x = (((x - self.a) * (self.max - self.min)) / (self.b - self.a)) + self.min
        return unscaled_x
