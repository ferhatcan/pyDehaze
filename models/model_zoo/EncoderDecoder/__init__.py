from .architectures import *

__all__ = ['EncoderDecoder_v01']

def make_model(args):
    if args.type == 'v01':
        return EncoderDecoder_v01(args)
    else:
        raise NotImplementedError('There is no such an architecture in Encoder-Decoder Network: {:}'.format(args.type))
