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
        self.argsLoss = ParseLoss(self.config)
        self.argsOptim = ParseOptimization(self.config)
        self.argsBenchs = ParseBenchmark(self.config)
        self.argsExperiment = ParseExperiment(self.config)


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
        self.desired_input_shape_multiplier = [int(v.strip()) for v in config["MODEL"]["desired_input_shape_multiplier"].split(',')]
        if self.generateNew:
            self.experiment_save_path = "runs/" + self.model + "/" \
            + self.experiment_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-hour%H')
        else:
            self.experiment_save_path = "runs/" + self.model + "/" \
                                        + self.experiment_name


class ParseDataset(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseDataset, self).__init__(config)

        self.dataset_name = config["DATASET"]["dataset_name"] if "dataset_name" in config["DATASET"] else "OTS"

        self.train_set_paths = config["DATASET"]["train_set_paths"].split(',\n')
        self.test_set_paths = config["DATASET"]["test_set_paths"].split(',\n')
        self.max_dataset_size = int(config["DATASET"]["max_dataset_size"]) if "max_dataset_size" in config["DATASET"] else 1e12

        self.normalize = config["DATASET"]["normalize"]
        self.validation_size = float(config["DATASET"]["validation_size"])
        self.shuffle_dataset = bool(config["DATASET"]["shuffle_dataset"])

class ParseModel(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseModel, self).__init__(config)

        self.type = config['MODEL']['type']
        self.input_dim = int(config["MODEL"]["input_dim"])
        self.output_dim = int(config["MODEL"]["output_dim"])
        self.include_input_image = bool(config['MODEL']['include_input_image']) if 'include_input_image' in config['MODEL'] \
                                                                                else False

class ParseLoss(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseLoss, self).__init__(config)

        self.loss = config['LOSS']['loss']

class ParseOptimization(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseOptimization, self).__init__(config)

        self.learning_rate      = float(config["OPTIMIZATION"]["learning_rate"])
        self.decay              = config["OPTIMIZATION"]["decay"]
        self.decay_factor_gamma = float(config["OPTIMIZATION"]["decay_factor_gamma"])
        self.optimizer          = config["OPTIMIZATION"]["optimizer"]
        self.momentum           = float(config["OPTIMIZATION"]["momentum"])
        self.betas              = list(map(float, config["OPTIMIZATION"]["betas"].split(',')))
        self.epsilon            = float(config["OPTIMIZATION"]["epsilon"])
        self.weight_decay       = float(config["OPTIMIZATION"]["weight_decay"])
        self.gclip              = float(config["OPTIMIZATION"]["gclip"])

class ParseBenchmark(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseBenchmark, self).__init__(config)

        self.benchmarks = config['BENCHMARK']['benchmarks']

class ParseExperiment(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseExperiment, self).__init__(config)

        self.log_frequency = float(config['EXPERIMENT']['log_frequency'])
        self.validate_frequency = float(config['EXPERIMENT']['validate_frequency'])
        self.epoch_num = int(config['EXPERIMENT']['epoch_num'])
