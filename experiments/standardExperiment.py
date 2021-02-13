import time

import torch
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

        self.loss_types = self.method.getLossTypes()

        self.records = dict()
        self.records["model_state_dict"] = self.method.model.state_dict()
        self.records["optimizer"] = self.method.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.records["current_epoch"] = 0
        self.records["best_loss"] = 1e10

        self.runtime_statistics = dict()
        self.runtime_statistics['best_benchmark_results'] = [0] * len(self.benchmark["name"])
        self.runtime_statistics['benchmark_result_names'] = self.benchmark["name"]

    def train_one_epoch(self):
        self.logger.logText('Epoch {}: \n'.format(self.records["current_epoch"]), 'epoch_log', verbose=True)

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
            for index, loss in enumerate(losses):
                training_statistics['total_losses'][index] = training_statistics['total_losses'][index] + \
                                                             loss.cpu().detach().numpy()

            if training_statistics['current_batch_number'] % (self.log_frequency * self.dataLoaders["train"]) == 0:
                self._log_training_statistics(training_statistics)

            if training_statistics['current_batch_number'] % (self.validate_frequency * self.dataLoaders["train"]) == 0:
                avg_valid_loss, benchmark_results_list = self.validate_dataset(self.dataLoaders['validation'])
                avg_train_loss = [x / training_statistics['current_batch_number'] for x in training_statistics['total_losses']]
                self._txtLogs('epoch_results', avg_train_loss, avg_valid_loss, benchmark_results_list)
                self._compareValidationResults(benchmark_results_list)

            training_statistics['current_batch_number'] += 1
            dataTimer = time.time()


        self.lr_scheduler.step()
        average_losses = [x / training_statistics['current_batch_number'] for x in training_statistics['total_losses']]
        return average_losses

    def train(self):
        self.load("model_last")
        if self.records["current_epoch"] == 0:
            self.logger.logText('----------------------------\nTraining Starting...\n', 'epoch_log', verbose=True)

        for epoch in range(self.records["current_epoch"], self.max_epoch):
            avg_train_loss = self.train_one_epoch()
            avg_valid_loss, benchmark_results_list = self.validate_dataset(dataloader=self.dataLoaders['validation'])
            self.records["current_epoch"] += 1

            self.logger.logText('----------------------------\nTraining Results...\n', 'epoch_results', verbose=True)
            self._txtLogs('epoch_results', avg_train_loss, avg_valid_loss, benchmark_results_list)
            self.save("model_last")

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
            for index, loss in enumerate(statistics['losses']):
                statistics['total_losses'][index] = validation_statistics['total_losses'][index] + \
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
        statistics['bench_results'] = [] * len(self.benchmark['name'])

        dataTimer = time.time()
        # Data load part
        data["inputs"] = data["inputs"].to(self.device)
        data["gts"] = data["gts"].to(self.device)
        statistics['dataload_duration'] += time.time() - dataTimer

        # Batch training part
        model_timer = time.time()
        result, losses = self.method.validation(data)
        data['result'] = result
        statistics['losses'] = losses
        statistics['model_duration'] += time.time() - model_timer

        for index, benchmark_method in enumerate(self.benchmark['bench_methods']):
            benchmark_result = benchmark_method(data)
            statistics['bench_results'][index] = benchmark_result

        return statistics

    def inference(self, image: torch.tensor):
        """
        :param image: 3D numpy image HeightxWidthxChannel
        :return:
        """
        assert len(image.shape) == 3, 'Input image tensor should be 3D tensor'
        image = image.transpose(2, 0, 1).to(self.device).unsqueeze(dim=0) # convert to 4D tensor
        result = self.method.test({'inputs': image})
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

    def _log_training_statistics(self, training_statistics):
        log_txt = '[Batch {}/{}]:\t'.format(training_statistics['current_batch_number'],
                                           training_statistics['total_batch_number'])

        for index, loss in enumerate(training_statistics['total_losses']):
            loss = loss / training_statistics['current_batch_number']
            load_time = training_statistics['total_dataload_duration'] / training_statistics['current_batch_number']
            model_time = training_statistics['total_model_duration'] / training_statistics['current_batch_number']
            log_txt += '{}: {:.4f}\t'.format(training_statistics['loss_names'][index], loss)
            log_txt += 'Data load duration: {:.2f} msec, Model duration: {:.2f} msec'.format(load_time, model_time)

        self.logger.logText(log_txt, 'epoch_results', verbose=True)

        avg_combined_loss = training_statistics['total_losses'][-1] / training_statistics['current_batch_number']
        self.records['best_loss'] = avg_combined_loss if avg_combined_loss < self.records['best_loss'] \
                               else self.records['best_loss']

    def _compareValidationResults(self, benchmark_results_list):
        index = 0
        for curr_bench_result, best_bench_result in zip(benchmark_results_list, self.runtime_statistics['best_benchmark_results']):
            if curr_bench_result["compare_func"](curr_bench_result, best_bench_result):
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