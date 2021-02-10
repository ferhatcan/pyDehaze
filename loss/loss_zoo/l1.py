import torch.nn as nn

from loss.ILoss import ILoss

class L1LossLocal(ILoss):
    def __init__(self, args):
        super(L1LossLocal, self).__init__(args)

        self.loss_function = nn.L1Loss()

    def forward(self, x: dict) -> list:
        assert "gts" in x and "result" in x, "inputs Type should be a dict and contains \"inputs\" and \"result\" keys"
        assert len(x["result"]) == 1, "there should be 1 result to calculate loss"
        result = []
        for i in range(len(x["gts"])):
            if x["gts"][i].shape == x["result"][0].shape:
                result.append(self.loss_function(x["gts"][i], x["result"][0]))
        return result

# @TODO EMRE TEST