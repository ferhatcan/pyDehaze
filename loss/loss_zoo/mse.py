import torch
import torch.nn as nn

from loss.ILoss import ILoss

class MSELossLocal(ILoss):
    def __init__(self, args):
        super(MSELossLocal, self).__init__(args)

        self.loss_function = nn.MSELoss()

    def forward(self, x: dict) -> list:
        assert "gts" in x and "result" in x, "gts Type should be a dict and contains \"gts\" and \"result\" keys"
        assert len(x["result"]) == 1, "there should be 1 result to calculate loss"
        result = []
        # only calculate with visible image
        for i in range(0, len(x["gts"])):
            if x["gts"][i].shape == x["result"][0].shape:
                result.append(self.loss_function(x["gts"][i], x["result"][0]))
        return result

# @TODO EMRE