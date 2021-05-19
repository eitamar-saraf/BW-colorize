import pytorch_lightning as pl

from models.blocks import Conv


class Encoder(pl.LightningModule):
    def __init__(self, in_c):
        super(Encoder, self).__init__()
        self.down1 = Conv(in_c, 32)
        self.down2 = Conv(32, 64)
        self.down3 = Conv(64, 128)
        self.down4 = Conv(128, 256)
        self.down5 = Conv(256, 512)
        self.down6 = Conv(512, 512)

        self.m = Conv(512, 512)

    def forward(self, x):
        # layer 1 (b, 1, 224, 224) -> (b, 32, 112, 112)
        down1 = self.down1(x)
        # layer 2 (b, 32, 112, 112) -> (b, 64, 56, 56)
        down2 = self.down2(down1)
        # layer 3 (b, 64, 56, 56) -> (b, 128, 28, 28)
        down3 = self.down3(down2)
        # layer 4 (b, 128, 28, 28) -> (b, 256, 14, 14)
        down4 = self.down4(down3)
        # layer 5  (b, 256, 14, 14) -> (b, 512, 7, 7)
        down5 = self.down5(down4)
        # layer 6 (b, 512, 7, 7) -> (b, 512, 3, 3)
        down6 = self.down6(down5)

        return down1, down2, down3, down4, down5, down6

    def training_step(self, x):
        z = self.forward(x)
        return z
