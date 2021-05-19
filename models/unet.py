from torch import optim
from torch.nn import functional as F

import pytorch_lightning as pl

from models.decoder import Decoder
from models.encoder import Encoder


class Unet(pl.LightningModule):
    def __init__(self, in_c, out_c):
        super(Unet, self).__init__()

        self.encoder = Encoder(in_c)
        self.decoder = Decoder(out_c)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    def l1(self, ab_hat, ab):
        return F.l1_loss(ab_hat, ab)

    def training_step(self, train_batch, batch_idx):
        l, ab = train_batch

        ab_hat = self.forward(l)
        loss = self.l1(ab_hat, ab)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        l, ab = val_batch
        ab_hat = self.forward(l)
        loss = self.l1(ab_hat, ab)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
