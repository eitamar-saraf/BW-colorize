import pytorch_lightning as pl
from torch import nn

from models.blocks import Conv


class Encoder(pl.LightningModule):
    def __init__(self, in_c):
        super(Encoder, self).__init__()
        self.conv1 = Conv(in_c, 32)
        self.conv2 = Conv(32, 64)
        self.conv3 = Conv(64, 128)
        self.conv4 = Conv(128, 256)
        self.conv5 = Conv(256, 512)
        self.conv6 = Conv(512, 512)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        # layer 1 (b, 1, 224, 224) -> (b, 32, 224, 224)
        x1 = self.conv1(x)
        # layer 2 (b, 32, 112, 112) -> (b, 64, 112, 112)
        x2 = self.conv2(self.mp(x1))
        # layer 3 (b, 64, 56, 56) -> (b, 128, 56, 56)
        x3 = self.conv3(self.mp(x2))
        # layer 4 (b, 128, 28, 28) -> (b, 256, 28, 28)
        x4 = self.conv4(self.mp(x3))
        # layer 5  (b, 256, 14, 14) -> (b, 512, 14, 14)
        x5 = self.conv5(self.mp(x4))
        # layer 6 (b, 512, 7, 7) -> (b, 512, 7, 7)
        x6 = self.conv6(self.mp(x5))

        m = self.mp(x6)
        return x1, x2, x3, x4, x5, x6, m

    def training_step(self, x):
        z = self.forward(x)
        return z
