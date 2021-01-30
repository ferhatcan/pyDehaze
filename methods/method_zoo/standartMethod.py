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
        data["result"] = [result]
        # calculate loss
        losses = self._calculateLoss({'result': data['result'], 'gts': data['gts']})
        self._checkNaN(losses)
        # calculate gradients
        losses[-1].backward()

        # back-propagation
        self.optimizer.step()

    def validate(self, data):
        assert 'inputs' in data, 'Input dictionary does not have "inputs" key!'
        assert 'gts' in data, 'Input dictionary does not have "gts" key!'

        self.model.eval()
        # run model
        with torch.no_grad():
            result = self.model({'inputs': data["inputs"]})
        data["result"] = [result]
        # calculate loss
        losses = self._calculateLoss({'result': data['result'], 'gts': data['gts']})
        self._checkNaN(losses)

        return result, losses

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