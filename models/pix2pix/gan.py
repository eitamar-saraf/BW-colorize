import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn

from loss.dicriminator_loss import DiscriminatorLoss
from models.pix2pix.discriminator import Discriminator
from models.pix2pix.generator import Generator
from utils.image_handler import reverse_transform


class GAN(LightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)
        self.gan_loss = DiscriminatorLoss()
        self.criteria = nn.L1Loss()
        self.lambda1 = 100.
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

    def forward(self, l):
        return self.generator(l)

    def training_step(self, batch, batch_idx, optimizer_idx):
        l, ab = batch
        fake_color = self.generator(l)
        fake_image = torch.cat([l, fake_color], dim=1)
        real_image = torch.cat([l, ab], dim=1)

        # train generator
        if optimizer_idx == 0:
            fake_predications = self.discriminator(fake_image)
            g_loss = self.gan_loss(fake_predications, target_is_real=True)

            # pixel-wise loss
            l1_loss = self.criteria(fake_color, ab) * self.lambda1

            # total loss
            g_loss += l1_loss
            self.log("Loss/generator", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            real_predications = self.discriminator(real_image)
            loss_discriminator_real = self.gan_loss(real_predications, target_is_real=True)

            fake_predications = self.discriminator(fake_image.detach())
            loss_discriminator_fake = self.gan_loss(fake_predications, target_is_real=False)

            # discriminator loss is the average of these
            d_loss = (loss_discriminator_real + loss_discriminator_fake) / 2

            self.log("Loss/discriminator", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        l, ab = batch
        fake_color = self.generator(l)
        fake_image = torch.cat([l, fake_color], dim=1)
        real_image = torch.cat([l, ab], dim=1)

        fake_predications = self.discriminator(fake_image)
        real_predications = self.discriminator(real_image)

        loss_discriminator_real = self.gan_loss(real_predications, target_is_real=True)
        loss_discriminator_fake = self.gan_loss(fake_predications, target_is_real=False)
        d_loss = (loss_discriminator_real + loss_discriminator_fake) / 2

        g_loss = self.gan_loss(fake_predications, target_is_real=True)
        l1_loss = self.criteria(fake_color, ab) * self.lambda1  # pixel-wise loss
        g_loss += l1_loss

        self.log("Loss/Validation", {'discriminator': d_loss, 'generator': g_loss}, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return {'fake_image': fake_image, 'real_image': real_image}

    def validation_epoch_end(self, outputs):

        first_sample = outputs[0]
        second_sample = outputs[1]

        for i, sample in enumerate([first_sample, second_sample]):
            fake_images = sample['fake_image'][:8]
            real_images = sample['real_image'][:8]

            # transpose to (C, H, W) for cv2
            fake_image = fake_images.permute(0, 2, 3, 1)
            real_image = real_images.permute(0, 2, 3, 1)
            # transform to numpy
            fake_image = fake_image.cpu().numpy()
            real_image = real_image.cpu().numpy()

            images = []
            for real, fake in zip(real_image, fake_image):
                real = reverse_transform(real)
                fake = reverse_transform(fake)

                # expand the dimension to (N, H, W, C) and concat on the row dimension
                sample = np.stack((real, fake))
                images.append(sample)

            images = np.vstack(images)
            self.logger.experiment.add_images(f"validation image {self.current_epoch}/batch {i}", images,
                                              global_step=self.global_step, dataformats='NHWC')

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    @torch.no_grad()
    def init_weights(self, m, gain=0.02):
        """weight initialisation of the different layers of the Generator and Discriminator"""
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
