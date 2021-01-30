import loss.loss_zoo as zoo

def getMSELoss(args):
    return zoo.MSELossLocal(args)

def getL1Loss(args):
    return zoo.L1LossLocal(args)