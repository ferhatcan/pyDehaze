import torch
import torch.nn as nn


class IBenchmark(nn.Module):
    def __init__(self, args):
        super(IBenchmark, self).__init__()
        self.args = args
        self.benchmarkDict = dict()

    def forward(self, data: dict) -> torch.tensor:
        """
        :param data: 'result' and 'gts' keys required
                    tensors should be in range [0, 255]
        :return: bench_dict -->
                    'result':
                    'bench_name':
                    'compare_func':
                    'max_value':
                    'min_value':
                    'describe_text':
        """
        raise NotImplementedError

    def fillBenchmarkDict(self, result):
        raise NotImplementedError

    @staticmethod
    def checkInput(data):
        assert 'result' and 'gts' in data, 'Benchmark input should include inputs and gts keys.'
        assert data['result'].shape == data['gts'].shape, 'input and ground truth tensor shape should be equal.'