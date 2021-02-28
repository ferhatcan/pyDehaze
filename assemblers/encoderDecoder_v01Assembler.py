import torch

from dataloaders.dataloaderGetter import *
from models.modelGetters import getEncoderDecoderModel
from loss.lossGetters import getMSELoss, getL1Loss
from optimizers.optimizerSchedulerGetter import *
from methods.methodGetter import getStandartMethod
from benchmarks.benchmarkGetter import getBenchmarks
from utils.logger import LoggerTensorBoard
from experiments.experimentGetter import getStandardExperiment


def getExperiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")

    print("The system will use following resource: {:}".format(args.argsCommon.device))
    print('Current available device is {}'.format(device))

    possibles = globals().copy()

    dataloaders = possibles.get('get' + args.argsDataset.dataset_name + 'Dataloaders')(args.argsDataset)

    model = getEncoderDecoderModel(args.argsModel)
    model.to(device)

    loss_dict = dict()
    loss_dict["types"] = []
    loss_dict["functions"] = []
    loss_dict["weights"] = []
    for loss in args.argsLoss.loss.split('+'):
        weight, loss_type = loss.split('*')
        loss_dict["functions"].append(possibles.get('get'+loss_type+'Loss')(args.argsLoss))
        loss_dict["weights"].append(float(weight))
        loss_dict["types"].append(loss_type)
    loss_dict["types"].append('total')

    lr_scheduler, optimizer = possibles.get('get'+args.argsOptim.optimizer+'Optimizer')(model.parameters(), args.argsOptim)

    method = getStandartMethod(model, loss_dict, optimizer, args.argsCommon)

    logger = LoggerTensorBoard(args.argsCommon.experiment_save_path, args.argsCommon.experiment_save_path + '/tensorboard')

    benchmarks = getBenchmarks(args.argsBenchs)

    experiment = getStandardExperiment(dataloaders, method, lr_scheduler, benchmarks, logger, args.argsExperiment)

    return experiment
