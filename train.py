from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

CONFIG_FILE_NAME = './configs/EncoderDecoder_v02.ini'

def main():
    experiment = getExperimentWithDesiredAssembler(CONFIG_FILE_NAME)
    experiment.train()

if __name__ == '__main__':
    main()