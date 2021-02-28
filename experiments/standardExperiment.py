import time

import torch
import numpy as np
from texttable import Texttable

from experiments.IExperiment import IExperiment

class StandardExperiment(IExperiment):
    def __init__(self,
                 dataLoaders,
                 method,
                 lr_scheduler,
                 benchmark,
                 logger,
                 args):
        super(StandardExperiment, self).__init__(args)

        self.dataLoaders = dataLoaders
        self.method = method
        self.lr_scheduler = lr_scheduler
        self.benchmark = benchmark
        self.logger = logger

        self.log_frequency = args.log_frequency
        self.validate_frequency = args.validate_frequency
        self.max_epoch = args.epoch_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        self.desired_input_shape_multiplier = args.desired_input_shape_multiplier

        self.loss_types = self.method.getLossTypes()

        self.records = dict()
        self.records["model_state_dict"] = self.method.model.state_dict()
        self.records["optimizer"] = self.method.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.records["current_epoch"] = 0
        self.records["best_loss"] = 1e10

        self.runtime_statistics = dict()
        self.runtime_statistics['best_benchmark_results'] = [None] * len(self.benchmark["name"])
        self.runtime_statistics['benchmark_result_names'] = self.benchmark["name"]

    def train_one_epoch(self):
        self.logger.logText('-----> Epoch {}: \n'.format(self.records["current_epoch"]), 'epoch_results', verbose=True)
        self.logger.logText('-----> Epoch {}: \n'.format(self.records["current_epoch"]), 'epoch_validation_results', verbose=False)

        training_statistics = dict()
        training_statistics['total_losses'] = [0] * len(self.loss_types)
        training_statistics['loss_names'] = self.loss_types
        training_statistics['current_batch_number'] = 1
        training_statistics['total_batch_number'] = len(self.dataLoaders["train"])
        training_statistics['total_dataload_duration'] = 0
        training_statistics['total_model_duration'] = 0

        dataTimer = time.time()
        for data in self.dataLoaders["train"]:
            # Data load part
            data["inputs"] = data["inputs"].to(self.device)
            data["gts"] = data["gts"].to(self.device)
            training_statistics['total_dataload_duration'] += time.time() - dataTimer

            # Batch training part
            model_timer = time.time()
            _, losses = self.method.train(data)
            training_statistics['total_model_duration'] += time.time() - model_timer

            # Filling statistics and logging part
            for index, loss in enumerate(losses['loss_values']):
                training_statistics['total_losses'][index] = training_statistics['total_losses'][index] + \
                                                             loss.cpu().detach().numpy()

            if training_statistics['current_batch_number'] % int(self.log_frequency * len(self.dataLoaders["train"]) + 1) == 0:
                self._log_training_statistics(training_statistics)

            if training_statistics['current_batch_number'] % int(self.validate_frequency * len(self.dataLoaders["train"]) + 1) == 0:
                avg_valid_loss, benchmark_results_list = self.validate_dataset(self.dataLoaders['validation'])
                avg_train_loss = [x / training_statistics['current_batch_number'] for x in training_statistics['total_losses']]
                self.logger.logText('################  Validation Starting  ################\n', 'epoch_validation_results', verbose=True)
                self._txtLogs('epoch_validation_results', avg_train_loss, avg_valid_loss, benchmark_results_list)
                self.logger.logText('################  Validation Finished  ################\n',
                                    'epoch_validation_results', verbose=True)
                self._compareValidationResults(benchmark_results_list)

            training_statistics['current_batch_number'] += 1
            dataTimer = time.time()


        self.lr_scheduler.step()
        average_losses = [x / training_statistics['current_batch_number'] for x in training_statistics['total_losses']]
        return average_losses

    def reset_previous_results(self):
        self.logger.resetText('epoch_results')
        self.logger.resetText('epoch_validation_results')

    def train(self):
        self.load("model_last")
        if self.records["current_epoch"] == 0:
            self.logger.logText('################  Training Starting  ################\n', 'epoch_results', verbose=True)

        for epoch in range(self.records["current_epoch"], self.max_epoch):
            avg_train_loss = self.train_one_epoch()
            avg_valid_loss, benchmark_results_list = self.validate_dataset(dataloader=self.dataLoaders['validation'])
            self.records["current_epoch"] += 1

            self.logger.logText('################  Training Epoch Results  ################\n', 'epoch_validation_results',
                                verbose=True)
            self._txtLogs('epoch_validation_results', avg_train_loss, avg_valid_loss, benchmark_results_list)
            self.logger.logText('################  Epoch Finished  ################\n',
                                'epoch_validation_results', verbose=True)

            self.save("model_last")

        self.logger.logText('################  Training Finished  ################\n', 'epoch_results', verbose=True)

    def validate_dataset(self, dataloader):
        validation_statistics = dict()
        validation_statistics['total_losses'] = [0] * len(self.loss_types)
        validation_statistics['loss_names'] = self.loss_types
        validation_statistics['bench_names'] = self.benchmark['name']
        validation_statistics['bench_results'] = [dict()] * len(self.benchmark['name'])
        validation_statistics['total_bench_results'] = [0] * len(self.benchmark['name'])

        validation_statistics['current_batch_number'] = 1
        validation_statistics['total_batch_number'] = len(self.dataLoaders["train"])
        validation_statistics['total_dataload_duration'] = 0
        validation_statistics['total_model_duration'] = 0


        for data in dataloader:
            statistics = self.validate_single(data)

            validation_statistics['bench_results'] = statistics['bench_results']
            for i in range(len(statistics['bench_results'])):
                validation_statistics['total_bench_results'][i] += statistics['bench_results'][i]['result']
            # Filling statistics and logging part
            for index, loss in enumerate(statistics['losses']['loss_values']):
                validation_statistics['total_losses'][index] = validation_statistics['total_losses'][index] + \
                                                    loss.cpu().detach().numpy()

            validation_statistics['current_batch_number'] += 1
            validation_statistics['total_dataload_duration'] += statistics['dataload_duration']
            validation_statistics['total_model_duration'] += statistics['model_duration']

        average_losses = [x / validation_statistics['current_batch_number'] for x in validation_statistics['total_losses']]
        average_bench_results = validation_statistics['bench_results']
        for i in range(len(average_bench_results)):
            average_bench_results[i]['result'] =  validation_statistics['total_bench_results'][i] / \
                                               validation_statistics['current_batch_number']

        return average_losses, average_bench_results

    def validate_single(self, data: dict):
        statistics = dict()
        statistics['bench_results'] = [dict()] * len(self.benchmark['name'])

        dataTimer = time.time()
        # Data load part
        data["inputs"] = data["inputs"].to(self.device)
        data["gts"] = data["gts"].to(self.device)
        statistics['dataload_duration'] = time.time() - dataTimer

        # Batch training part
        model_timer = time.time()
        result, losses = self.method.validate(data)
        data['result'] = result
        statistics['losses'] = losses
        statistics['model_duration'] = time.time() - model_timer

        for index, benchmark_method in enumerate(self.benchmark['bench_methods']):
            # @TODO there is an error here (input data should be normalized to 0-255)
            benchmark_result = benchmark_method(data)
            statistics['bench_results'][index] = benchmark_result

        return statistics

    def inference(self, image: torch.tensor):
        """
        :param image: 3D numpy image HeightxWidthxChannel
        :return:
        """
        assert len(image.shape) == 3, 'Input image array should be 3D numpy array'
        padded_img, _ = self.arrange_input_image(image)
        padded_img = ((padded_img.astype(np.float32) / 255) - 0.5) * 2
        padded_img = padded_img.transpose(2, 0, 1) # convert to 4D tensor
        padded_img = torch.from_numpy(padded_img).unsqueeze(dim=0).to(self.device)
        result = self.method.test({'inputs': padded_img})
        result = (result + 1) * 0.5
        return result.cpu().detach().squeeze().numpy().transpose(1, 2, 0)

    def save(self, saveName: str):
        self.records["model_state_dict"] = self.method.model.state_dict()
        self.records["optimizer"] = self.method.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()

        self.logger.saveCheckpoint(self.records, saveName)

    def load(self, loadName: str):
        state_dicts = self.logger.loadCheckpoint(loadName)
        if not state_dicts is None:
            if "model_state_dict" in state_dicts:
                self.records["model_state_dict"] = state_dicts["model_state_dict"]
            if "optimizer" in state_dicts:
                self.records["optimizer"] = state_dicts["optimizer"]
            if "lr_scheduler" in state_dicts:
                self.records["lr_scheduler"] = state_dicts["lr_scheduler"]
            if "current_epoch" in state_dicts:
                self.records["current_epoch"] = state_dicts["current_epoch"]

            self.method.model.load_state_dict(self.records["model_state_dict"])
            self.method.optimizer.load_state_dict(self.records["optimizer"])
            self.lr_scheduler.load_state_dict(self.records["lr_scheduler"])

    def arrange_input_image(self, image):
        padding_dimensions = []
        for i, v in enumerate(self.desired_input_shape_multiplier):
            padding = (v - (image.shape[i] % int(v))) % int(v)
            padding_dimensions.append((padding // 2, padding - padding // 2))
        padding_dimensions.append((0,0))
        image_padded = np.pad(image, padding_dimensions, 'constant')

        return image_padded, padding_dimensions

    def _log_training_statistics(self, training_statistics):
        log_txt = '[Batch {}/{}]:\t'.format(training_statistics['current_batch_number'],
                                           training_statistics['total_batch_number'])

        for index, loss in enumerate(training_statistics['total_losses']):
            loss = loss / training_statistics['current_batch_number']
            log_txt += '{}: {:.5f}\t'.format(training_statistics['loss_names'][index], loss)
        load_time = training_statistics['total_dataload_duration'] / training_statistics['current_batch_number']
        model_time = training_statistics['total_model_duration'] / training_statistics['current_batch_number']
        log_txt += 'Data load duration: {:.2f} msec, Model duration: {:.2f} msec\n'.format(load_time, model_time)

        self.logger.logText(log_txt, 'epoch_results', verbose=True)

        avg_combined_loss = training_statistics['total_losses'][-1] / training_statistics['current_batch_number']
        self.records['best_loss'] = avg_combined_loss if avg_combined_loss < self.records['best_loss'] \
                               else self.records['best_loss']

    def _compareValidationResults(self, benchmark_results_list):
        index = 0
        for curr_bench_result, best_bench_result in zip(benchmark_results_list, self.runtime_statistics['best_benchmark_results']):
            if not best_bench_result is None:
                if curr_bench_result["compare_func"](curr_bench_result['result'], best_bench_result['result']):
                    self.runtime_statistics['best_benchmark_results'][index] = curr_bench_result
                    self.save('model_best_' + curr_bench_result['bench_name'])
                    self.save('model_best')
            else:
                self.runtime_statistics['best_benchmark_results'][index] = curr_bench_result
                self.save('model_best_' + curr_bench_result['bench_name'])
                self.save('model_best')
            index += 1

    def _txtLogs(self, fileName, training_loss, validation_loss, validation_benchmark_list):
        t = Texttable()
        t.add_rows([['**losses**'] + self.loss_types,
                    ['training'] + training_loss,
                    ['validation'] + validation_loss])
        self.logger.logText(t.draw() + '\n', fileName, verbose=True)

        validation_benchmark_results = [bench['result'] for bench in validation_benchmark_list]

        t = Texttable()
        t.add_rows([['**benchs**'] + self.benchmark["name"],
                    ['results'] + validation_benchmark_results])
        self.logger.logText(t.draw() + '\n', fileName, verbose=True)


# if __name__ == '__main__':
#     from dataloaders.dataloaderGetter import getOTSDataloaders
#     from models.modelGetters import getEncoderDecoderModel
#     from loss.lossGetters import getMSELoss, getL1Loss
#     from optimizers.optimizerSchedulerGetter import *
#     from methods.methodGetter import getStandartMethod
#     from benchmarks.benchmarkGetter import getBenchmarks
#     from utils.logger import LoggerTensorBoard
#
#     from utils.configParser import options
#
#     CONFIG_FILE_NAME = '../configs/EncoderDecoder_v01.ini'
#     args = options(CONFIG_FILE_NAME)
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")
#
#     dataloaders = getOTSDataloaders(args.argsDataset)
#
#     model = getEncoderDecoderModel(args.argsModel)
#     model.to(device)
#
#     possibles = globals().copy()
#     loss_dict = dict()
#     loss_dict["types"] = []
#     loss_dict["functions"] = []
#     loss_dict["weights"] = []
#     for loss in args.argsLoss.loss.split('+'):
#         weight, loss_type = loss.split('*')
#         loss_dict["functions"].append(possibles.get('get'+loss_type+'Loss')(args.argsLoss))
#         loss_dict["weights"].append(float(weight))
#         loss_dict["types"].append(loss_type)
#     loss_dict["types"].append('total')
#
#     lr_scheduler, optimizer = possibles.get('get'+args.argsOptim.optimizer+'Optimizer')(model.parameters(), args.argsOptim)
#
#     method = getStandartMethod(model, loss_dict, optimizer, args.argsCommon)
#
#     logger = LoggerTensorBoard(args.argsCommon.experiment_save_path, args.argsCommon.experiment_save_path + '/tensorboard')
#
#     benchmarks = getBenchmarks(args.argsBenchs)
#
#     experiment = StandardExperiment(dataloaders, method, lr_scheduler, benchmarks, logger, args.argsExperiment)
#
#     experiment.reset_previous_results()
#     # experiment.train()
#     experiment.load('model_best')
#     tmp = experiment.validate_dataset(dataloaders['validation'])
#
#     import PIL.Image as Image
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     img_path = 'D:/Image_Datasets/dehaze/RTTS/RTTS/JPEGImages/BD_Baidu_256.png'
#     img = np.array(Image.open(img_path).convert('RGB'))
#     img = experiment.arrange_input_image(img)
#     haze_diff = experiment.inference(img)
#
#     haze_diff = (haze_diff * 255).astype(np.uint8)
#     result = (img - haze_diff)
#
#     plt.figure()
#     plt.imshow(haze_diff)
#     plt.waitforbuttonpress()
#
#
#     plt.figure()
#     plt.imshow(result.astype(np.uint8))
#     plt.waitforbuttonpress()
#
#
#     tmp = 0