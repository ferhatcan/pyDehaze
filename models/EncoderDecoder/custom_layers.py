import torch.nn as nn

from models.IModel import IModel


class Encoder(IModel):
    def __init__(self, args):
        super(Encoder, self).__init__(args)

        self.encoder = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.encoder(x)

        return out


class Decoder(IModel):
    def __init__(self, args):
        super(Decoder, self).__init__(args)

        self.decoder = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=(1, 1), groups=32, bias=False)

    def forward(self, x):
        out = self.decoder(x)

        return out