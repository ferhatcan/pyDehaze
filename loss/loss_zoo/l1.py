import torch.nn as nn

from loss.ILoss import ILoss

class L1LossLocal(ILoss):
    def __init__(self, args):
        super(L1LossLocal, self).__init__(args)

        self.loss_function = nn.L1Loss()

    def forward(self, x: dict) -> list:
        assert "gts" in x and "result" in x, "inputs Type should be a dict and contains \"inputs\" and \"result\" keys"
        result = []
        if x["gts"].shape == x["result"].shape:
            result.append(self.loss_function(x["gts"], x["result"]))
        return result

# @TODO EMRE TEST