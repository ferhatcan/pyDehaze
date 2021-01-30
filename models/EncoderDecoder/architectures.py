import torch.nn as nn

from models.IModel import IModel
from models.EncoderDecoder.custom_layers import Encoder, Decoder

class EncoderDecoder_v01(IModel):
    def __init__(self, args):
        super(EncoderDecoder_v01, self).__init__(args)

        self.encoder1 = Encoder(args)
        self.encoder2 = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        out1 = self.encoder1(x['inputs'])
        out2 = self.encoder2(x)


        out = self.decoder({'layers': [out1, out2]})

        return out