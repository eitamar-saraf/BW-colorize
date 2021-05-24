import pytorch_lightning as pl

from models.blocks import Conv, DeConv


class Decoder(pl.LightningModule):
    def __init__(self, out_c):
        super(Decoder, self).__init__()
        self.m = Conv(512, 512)

        self.deconv6 = DeConv(1024, 512, padding=0, output_padding=0)
        self.deconv5 = DeConv(1024, 256)
        self.deconv4 = DeConv(512, 128)
        self.deconv3 = DeConv(256, 64)
        self.deconv2 = DeConv(128, 32)
        self.deconv1 = DeConv(64, out_c)

    def forward(self, z):
        skip_x1, skip_x2, skip_x3, skip_x4, skip_x5, skip_x6, m = z

        # layer 1  (b, 512, 3, 3) ->  (b, 512, 3, 3)
        m = self.m(m)

        # layer 2  (b, 512, 3, 3), (512, 7, 7) ->  (b, 512, 7, 7)
        x6 = self.deconv6(m, skip_x6)

        # layer 3  (b, 512, 7, 7), (b, 512, 14, 14) ->  (b, 256, 14, 14)
        x5 = self.deconv5(x6, skip_x5)

        # layer 4  (b, 256, 14, 14), (b, 256, 28, 28) ->  (b, 128, 28, 28)
        x4 = self.deconv4(x5, skip_x4)

        # layer 5  (b, 128, 28, 28), (b, 128, 56, 56) ->  (b, 64, 56, 56)
        x3 = self.deconv3(x4, skip_x3)

        # layer 6  (b, 64, 56, 56), (b, 64, 112, 112) ->  (b, 32, 112, 112)
        x2 = self.deconv2(x3, skip_x2)

        # layer 7  (b, 32, 112, 112), (b, 32, 224, 224) ->  (b, 2, 224, 224)
        x1 = self.deconv1(x2, skip_x1)

        return x1
