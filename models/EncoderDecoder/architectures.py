import torch.nn as nn

from models.IModel import IModel
from models.EncoderDecoder.custom_layers import Encoder, Decoder


class EncoderDecoder_v01(IModel):
    def __init__(self, args):
        super(EncoderDecoder_v01, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, inputs: list):
        assert len(inputs) == 1, 'There should be 1 input'
        feats = self.encoder(inputs[0])
        out = self.decoder(feats)
        return out

# @TODO EMRE TEST