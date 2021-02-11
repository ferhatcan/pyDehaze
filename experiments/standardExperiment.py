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

        self.max_epoch = args.epoch_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

    def train_one_epoch(self):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.max_epoch):
            avg_train_loss = self.train_one_epoch()
            avg_valid_loss, benchmark_results = self.test_dataset(dataloader=self.dataloaders['validation'])

    def test_dataset(self, dataloader):
        raise NotImplementedError

    def test_single(self, images: list):
        raise NotImplementedError

    def inference(self, data: dict):
        raise NotImplementedError

    def save(self, saveName: str):
        raise NotImplementedError

    def load(self, loadName: str):
        raise NotImplementedError