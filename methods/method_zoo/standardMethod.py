import torch
import numpy as np
import torch.nn as nn

from methods.IMethod import IMethod

class StandartMethod(IMethod):
    def __init__(self, model: nn.Module, loss_functions: dict, optimizer, args):
        super(StandartMethod, self).__init__(model, loss_functions, optimizer, args)

    def train(self, data):
        assert 'inputs' in data, 'Input dictionary does not have "inputs" key!'
        assert 'gts' in data, 'Input dictionary does not have "gts" key!'

        self.model.train()
        # clear gradients
        self.optimizer.zero_grad()
        # run model
        result = self.model({'inputs': data["inputs"]})
        data["result"] = result
        # calculate loss
        losses = self._calculateLoss({'result': data['result'], 'gts': data['gts']})
        self._checkNaN(losses)
        # calculate gradients
        losses[-1].backward()

        # back-propagation
        self.optimizer.step()

        return result, {'loss_values': losses, 'loss_types': self.loss_functions['types']}

    def validate(self, data):
        assert 'inputs' in data, 'Input dictionary does not have "inputs" key!'
        assert 'gts' in data, 'Input dictionary does not have "gts" key!'

        self.model.eval()
        # run model
        with torch.no_grad():
            result = self.model({'inputs': data["inputs"]})
        data["result"] = result
        # calculate loss
        losses = self._calculateLoss({'result': data['result'], 'gts': data['gts']})
        self._checkNaN(losses)

        return result, {'loss_values': losses, 'loss_types': self.loss_functions['types']}

    def test(self, data):
        assert 'inputs' in data, 'Input dictionary does not have "inputs" key!'

        self.model.eval()
        # run model
        with torch.no_grad():
            result = self.model({'inputs': data["inputs"]})

        return result

    def _calculateLoss(self, inputs):
        losses = []
        for weight, loss_function in zip(self.loss_functions['weights'], self.loss_functions['functions']):
            results = loss_function(inputs)
            losses.append(weight * sum(results) / len(results))
        losses.append(sum(losses))
        return losses

    def _checkNaN(self, inputs):
        for i, inp in enumerate(inputs):
            if np.isnan(inp.cpu().detach().numpy()):
                raise ValueError('Loss function returns NaN for loss type {}'.format(self.loss_functions['types'][i]))


# if __name__ == '__main__':
#     from models.modelGetters import getEncoderDecoderModel
#     from loss.lossGetters import getMSELoss, getL1Loss
#     from optimizers.optimizerSchedulerGetter import *
#
#     from dataloaders.dataloaderGetter import getOTSDataloaders
#     from utils.configParser import options
#
#     CONFIG_FILE_NAME = "../../configs/EncoderDecoder_v01.ini"
#     args = options(CONFIG_FILE_NAME)
#
#     model = getEncoderDecoderModel(args.argsModel)
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
#     lr_scheduler, optimizer = possibles.get('get' + args.argsOptim.optimizer + 'Optimizer')(model.parameters(),
#                                                                                              args.argsOptim)
#
#     dataloaders = getOTSDataloaders(args.argsDataset)
#     data = next(iter(dataloaders["train"]))
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.argsCommon.device == "gpu" else "cpu")
#     data["inputs"] = data["inputs"].to(device)
#     data["gts"] = data["gts"].to(device)
#
#     model.to(device)
#
#     method = StandartMethod(model, loss_dict, optimizer, None)
#
#     out_train = method.train(data)
#     out_eval = method.validate(data)
#     out_test = method.test(data)
#     tmp = 0