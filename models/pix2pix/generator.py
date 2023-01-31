from torch import nn
import torch


class Generator(nn.Module):
    """Generator of the Pix2Pix model.
       For the Lab version, nb_output_channels=2"""

    def __init__(self, nb_output_channels=2):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_4 = nn.BatchNorm2d(512)
        self.conv2d_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_5 = nn.BatchNorm2d(512)
        self.conv2d_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_6 = nn.BatchNorm2d(512)
        self.conv2d_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2d_8 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                                           bias=False, output_padding=1)
        self.batchnorm_8 = nn.BatchNorm2d(512)
        self.conv2d_9 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1,
                                           bias=False, output_padding=1)
        self.batchnorm_9 = nn.BatchNorm2d(512)
        self.conv2d_10 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_10 = nn.BatchNorm2d(512)
        self.conv2d_11 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=256, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_11 = nn.BatchNorm2d(256)
        self.conv2d_12 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_12 = nn.BatchNorm2d(128)
        self.conv2d_13 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_13 = nn.BatchNorm2d(64)
        self.conv2d_14 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=nb_output_channels, kernel_size=4,
                                            stride=2, padding=1, bias=True)

    def forward(self, encoder_input):
        # encoder
        encoder_output_1 = self.leakyrelu(self.conv2d_1(encoder_input))
        encoder_output_2 = self.leakyrelu(self.batchnorm_2(self.conv2d_2(encoder_output_1)))
        encoder_output_3 = self.leakyrelu(self.batchnorm_3(self.conv2d_3(encoder_output_2)))
        encoder_output_4 = self.leakyrelu(self.batchnorm_4(self.conv2d_4(encoder_output_3)))
        encoder_output_5 = self.leakyrelu(self.batchnorm_5(self.conv2d_5(encoder_output_4)))
        encoder_output_6 = self.leakyrelu(self.batchnorm_6(self.conv2d_6(encoder_output_5)))
        encoder_output = self.conv2d_7(encoder_output_6)
        # decoder
        decoder_output = self.dropout(self.batchnorm_8(self.conv2d_8(self.relu(encoder_output))))
        decoder_output = self.dropout(self.batchnorm_9(
            self.conv2d_9(self.relu(torch.cat([encoder_output_6, decoder_output], 1)))))  # skip connection
        decoder_output = self.batchnorm_10(
            self.conv2d_10(self.relu(torch.cat([encoder_output_5, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_11(
            self.conv2d_11(self.relu(torch.cat([encoder_output_4, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_12(
            self.conv2d_12(self.relu(torch.cat([encoder_output_3, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_13(
            self.conv2d_13(self.relu(torch.cat([encoder_output_2, decoder_output], 1))))  # skip connection
        decoder_output = self.activation(
            self.conv2d_14(self.relu(torch.cat([encoder_output_1, decoder_output], 1))))  # skip connection
        return decoder_output
