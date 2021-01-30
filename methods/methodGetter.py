import methods.method_zoo as zoo

def getStandartMethod(model, loss_function, optimizer, args):
    return zoo.StandartMethod(model, loss_function, optimizer, args)