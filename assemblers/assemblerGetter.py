from utils.configParser import options

from assemblers.encoderDecoder_v01Assembler import getExperiment as encoderDecoder_v01

def getExperimentWithDesiredAssembler(config_file_name):
    args = options(config_file_name)
    if args.argsCommon.model == 'encoderDecoder':
        if args.argsModel.type == 'v01' or args.argsModel.type == 'v02' \
                or args.argsModel.type == 'resnetv01' or args.argsModel.type == 'resnetv02'\
                or args.argsModel.type == 'resnetv01-DIV2K':
            return encoderDecoder_v01(args)
        else:
            ValueError('There should be an valid assembler type: given type {:}'.format(args.argsModel.type))
    else:
        ValueError('There should be an valid model: given {:}'.format(args.argsCommon.model))
