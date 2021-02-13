import os

import torch

from experiments.IExperiment import IExperiment

class StandardExperiment(IExperiment):
    def __init__(self,
                 model,
                 dataLoaders,
                 loss,
                 method,
                 optimizer,
                 lr_scheduler,
                 benchmark,
                 logger,
                 args):
        super(StandardExperiment, self).__init__(args)

        self.model = model
        self.dataloaders = dataLoaders
        self.loss = loss
        self.method = method
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.benchmark = benchmark
        self.logger = logger

        self.num_of_loss = len(self.loss["loss_types"])

        self.max_epoch = args.epoch_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        self.records = dict()
        self.records["model_state_dict"] = self.model.state_dict()
        self.records["optimizer"] = self.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        self.records["current_epoch"] = 0

        self.destination_checkpoint = ".pretrainednetworks"

    def train_one_epoch(self):
        total_losses = [0] * self.num_of_loss
        for data in self.dataloaders["train"]:
            data["inputs"] = data["inputs"].to(self.device)
            data["gts"] = data["gts"].to(self.device)
            _, losses = self.method.train(data)
            for index, loss in enumerate(losses):
                total_losses[index] = total_losses[index] + loss.cpu().detach().numpy()
        self.lr_scheduler.step()
        average_losses = [x / (len(self.dataloaders["train"])) for x in total_losses]
        return average_losses

    def train(self):
        self.load("model_last")
        for epoch in range(self.records["current_epoch"], self.max_epoch):
            avg_train_loss = self.train_one_epoch()
            avg_valid_loss, benchmark_results = self.test_dataset(dataloader=self.dataloaders['validation'])
            self.records["current_epoch"] += 1
            self.save("model_last")

    def test_dataset(self, dataloader):
        raise NotImplementedError

    def test_single(self, images: list):
        raise NotImplementedError

    def inference(self, data: dict):
        raise NotImplementedError

    def save(self, saveName: str):
        self.records["model_state_dict"] = self.model.state_dict()
        self.records["optimizer"] = self.optimizer.state_dict()
        self.records["lr_scheduler"] = self.lr_scheduler.state_dict()
        savePath = os.path.join(self.destination_checkpoint, saveName) + '.pth'
        torch.save(self.records, savePath)

    def load(self, loadName: str):
        loadPath = os.path.join(self.destination_checkpoint, loadName) + '.pth'
        if os.path.exists(loadPath):
            state_dicts = torch.load(loadPath)
            self.records["model_state_dict"] = state_dicts["model_state_dict"]
            self.records["optimizer"] = state_dicts["optimizer"]
            self.records["lr_scheduler"] = state_dicts["lr_scheduler"]
            self.records["current_epoch"] = state_dicts["current_epoch"]
            self.model.load_state_dict(self.records["model_state_dict"])
            self.optimizer.load_state_dict(self.records["optimizer"])
            self.lr_scheduler.load_state_dict(self.records["lr_scheduler"])