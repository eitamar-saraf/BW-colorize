import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    """for the patch discriminator, the output is a 30x30 tensor
       if the image is real, it should return all ones 'real_labels'
       if the image is fake, it should return all zeros 'fake_labels'
       returns the MSE loss between the output of the discriminator and the label"""

    def __init__(self):
        super().__init__()
        self.register_buffer('real_labels', torch.ones([26, 26], requires_grad=False), False)
        self.register_buffer('fake_labels', torch.zeros([26, 26], requires_grad=False), False)
        # use MSE loss for the discriminator
        self.loss = nn.MSELoss()

    def forward(self, predictions, target_is_real):
        if target_is_real:
            target = self.real_labels
        else:
            target = self.fake_labels
        return self.loss(predictions, target.expand_as(predictions))
