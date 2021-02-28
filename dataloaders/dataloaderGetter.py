import torch
import numpy as np

import dataloaders.dataset_zoo as zoo


def _generateDataLoaders(ds_train, ds_test, batch_size, max_dataset_size=4000, validation_percent=0.1, isShuffle=True) -> dict:
    dataset_size = len(ds_train)
    # indices = list(range(min(dataset_size, 10000)))
    indices = np.random.randint(dataset_size, size=(min(dataset_size, max_dataset_size))).tolist()
    split = int(np.floor(validation_percent * len(indices)))
    if isShuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    valid_ds = torch.utils.data.Subset(ds_train, val_indices)
    train_ds = torch.utils.data.Subset(ds_train, train_indices)

    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    # validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    #
    # loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    # loader_val = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=validation_sampler, num_workers=0)
    # loader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=1)
    loader_val = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=1)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=1)

    loaders = dict()
    loaders["train"] = loader_train
    loaders["validation"] = loader_val
    loaders["test"] = loader_test

    return loaders


def getOTSDataloaders(args):
    # np.random.seed(args.seed)
    ds_train = zoo.OTSDataset(args=args, train=True)
    ds_test = zoo.OTSDataset(args=args, train=False)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.max_dataset_size, args.validation_size, args.shuffle_dataset)

def getDIV2KDataloaders(args):
    # np.random.seed(args.seed)
    ds_train = zoo.DIV2KDataset(args=args, train=True)
    ds_test = zoo.DIV2KDataset(args=args, train=False)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.max_dataset_size, args.validation_size, args.shuffle_dataset)

def getNTIREDataloaders(args):
    # np.random.seed(args.seed)
    ds_train = zoo.NTIREDataset(args=args, train=True)
    ds_test = zoo.NTIREDataset(args=args, train=False)
    return _generateDataLoaders(ds_train, ds_test, args.batch_size, args.max_dataset_size, args.validation_size, args.shuffle_dataset)