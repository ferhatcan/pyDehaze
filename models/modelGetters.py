from models.model_zoo.EncoderDecoder import make_model as encoder_make_model


def getEncoderDecoderModel(args):
    return encoder_make_model(args)