from experiments.standardExperiment import StandardExperiment

def getStandardExperiment(dataloaders, method, lr_scheduler, benchmarks, logger, args):
    return StandardExperiment(dataloaders, method, lr_scheduler, benchmarks, logger, args)