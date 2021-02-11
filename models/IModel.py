import torch
import torch.nn as nn


class IModel(nn.Module):
    def __init__(self, args):
        super(IModel, self).__init__()
        self.args = args

    def forward(self, x: dict) -> torch.Tensor:
        """
        :param x: a dictionary containing all useful information for model
            ex: x = {'layers': skip connection inputs , 'input_image'}
            :warning required sanity check should be implemented in model
        :return:
        """
        raise NotImplementedError