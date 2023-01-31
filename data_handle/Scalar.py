class TorchStandardScaler:

    def __init__(self):
        self.b = 1
        self.a = -1
        self.min = 0
        self.max = 1

    def __call__(self, x):
        scaled_x = self.a + (x - self.min) * (self.b - self.a) / (self.max - self.min)
        return scaled_x

    def inverse_transform(self, x):
        unscaled_x = (((x - self.a) * (self.max - self.min)) / (self.b - self.a)) + self.min
        return unscaled_x
