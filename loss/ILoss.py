import torch
import torch.nn as nn

class ILoss(nn.Module):
    def __init__(self, args):
        super(ILoss, self).__init__()
        self.args = args

    def forward(self, x: dict) -> torch.Tensor:
        """
        :param x: dictionary containing result and gt keys
        :return: tensor total loss
        """
        raise NotImplementedError
