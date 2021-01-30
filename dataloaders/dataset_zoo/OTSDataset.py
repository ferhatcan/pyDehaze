import torch

from dataloaders.IDataset import IDataset


class OTSDataset(IDataset):
    def __init__(self, args, train=False):
        super(OTSDataset, self).__init__()

        self.image_paths = args.train_set_paths if train else args.test_set_paths
