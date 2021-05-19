import pytorch_lightning as pl
from torch import nn

from models.blocks import Conv, DeConv


class Decoder(pl.LightningModule):
    def __init__(self, out_c):
        super(Decoder, self).__init__()
        self.up6 = DeConv(256, 256)
        self.up6conv = Conv(512, 256)

        self.up5 = DeConv(256, 256)
        self.up5conv = Conv(512, 256)

        self.up4 = DeConv(256, 256)
        self.up4conv = Conv(512, 256)

        self.up3 = DeConv(256, 128)
        self.up3conv = Conv(256, 128)

        self.up2 = DeConv(128, 64)
        self.up2conv = Conv(128, 64)

        self.up1 = DeConv(64, 32)
        self.up1conv = Conv(64, 32)

        self.last_conv = nn.Conv2d(32, out_c, kernel_size=(1, 1))
        self.act_out = nn.Tanh()

    def forward(self, z):
        down1, down2, down3, down4, down5, down6 = z
        up6 = self.up6(down6, down6)  # up6 = (1024x6x6)
        up6conv = self.up6conv(up6)  # up6conv = (1024x6x6)

        up5 = self.up5(up6conv, down5)  # up5 = (1024x12x12)
        up5conv = self.up5conv(up5)  # up6conv = (512x12x12)

        up4 = self.up4(up5conv, down4)  # up4 = (512x24x24)
        up4conv = self.up4conv(up4)  # up4conv = (256x24x24)

        up3 = self.up3(up4conv, down3)  # up3 = (256x48x48)
        up3conv = self.up3conv(up3)  # up3conv = (128x48x48)

        up2 = self.up2(up3conv, down2)  # up2 = (128x96x96)
        up2conv = self.up2conv(up2)  # up2conv = (64x96x96)

        up1 = self.up1(up2conv, down1)  # up1 = (64x192x192)
        up1conv = self.up1conv(up1)  # up1conv = (32x192x192)

        pre_act = self.last_conv(up1conv)  # (2x192x192)
        out = self.act_out(pre_act)

        return down1, down2, down3, down4, down5, down6
