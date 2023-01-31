from torch import nn


class Discriminator(nn.Module):
    """Patch discriminator of the Pix2Pix model."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False)
        self.batchnorm_4 = nn.BatchNorm2d(512)
        self.conv2d_5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True)

    def forward(self, input):
        output = self.leakyrelu(self.conv2d_1(input))
        output = self.leakyrelu(self.batchnorm_2(self.conv2d_2(output)))
        output = self.leakyrelu(self.batchnorm_3(self.conv2d_3(output)))
        output = self.leakyrelu(self.batchnorm_4(self.conv2d_4(output)))
        output = self.sigmoid(self.conv2d_5(output))
        return output
