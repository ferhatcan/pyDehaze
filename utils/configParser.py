from configparser import ConfigParser
import importlib
import datetime

class options:
    def __init__(self, config_file_name):
        self.config = ConfigParser()
        self.config.read(config_file_name)

        self.argsCommon = ParseCommons(self.config)
        self.argsDataset = ParseDataset(self.config)
        self.argsModel = ParseModel(self.config)


class ParseCommons:
    def __init__(self, config: ConfigParser):
        self.experiment_name    = config["DEFAULT"]["experiment_name"]
        self.generateNew        = config["DEFAULT"].getboolean("generate_new_experiment")
        self.device             = config["HARDWARE"]["device"]
        self.seed               = int(config["HARDWARE"]["seed"])
        self.n_GPUs             = int(config["HARDWARE"]["n_GPUs"])
        self.precision          = config["HARDWARE"]["precision"]
        self.model              = config["MODEL"]["model"]
        self.input_shape           = list(map(int, config["DATASET"]["input_shape"].split(',')))
        self.batch_size         = int(config["DATASET"]["batch_size"])
        if self.generateNew:
            self.experiment_save_path = "runs/" + self.model + "/" \
            + self.experiment_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-hour%H')
        else:
            self.experiment_save_path = "runs/" + self.model + "/" \
                                        + self.experiment_name


class ParseDataset(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseDataset, self).__init__(config)

        self.train_set_paths = config["DATASET"]["train_set_paths"].split(',\n')
        self.test_set_paths = config["DATASET"]["test_set_paths"].split(',\n')

        self.normalize = config["DATASET"]["normalize"]
        self.validation_size = float(config["DATASET"]["validation_size"])
        self.shuffle_dataset = bool(config["DATASET"]["shuffle_dataset"])

class ParseModel(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseModel, self).__init__(config)

        self.input_dim = int(config["MODEL"]["input_dim"])
        self.output_dim = int(config["MODEL"]["output_dim"])
