import torch
import torch.nn as nn

from loss.ILoss import ILoss

class MSELossLocal(ILoss):
    def __init__(self, args):
        super(MSELossLocal, self).__init__(args)

        self.loss_function = nn.MSELoss()

    def forward(self, x: dict) -> list:
        assert "gts" in x and "result" in x, "gts Type should be a dict and contains \"gts\" and \"result\" keys"
        result = []
        if x["gts"].shape == x["result"].shape:
            result.append(self.loss_function(x["gts"], x["result"]))
        return result

# @TODO EMRE