from operator import lt

import torch

from benchmarks.IBenchmark import IBenchmark

class MSE(IBenchmark):
    def __init__(self, args):
        super(MSE, self).__init__(args)

    def forward(self, data: dict) -> torch.tensor:
        self.checkInput(data)

        inp, gts = self.normalizeInputs(data)
        result = torch.sqrt(torch.mean((inp - gts) ** 2))

        self.fillBenchmarkDict(result)

        return self.benchmarkDict

    def fillBenchmarkDict(self, result):
        self.benchmarkDict['bench_name'] = 'MSE'
        self.benchmarkDict['compare_func'] = lt
        self.benchmarkDict['max_value'] = 1e10
        self.benchmarkDict['min_value'] = 0
        self.benchmarkDict['describe_text'] = 'MSE score is {:.3f}'.format(result)
        self.benchmarkDict['result'] = result

    @staticmethod
    def normalizeInputs(data):
        if 'data_range' in data:
            data_range = data['data_range']
        else:
            data_range = (-1, 1)

        inp = (((data['result'] - data_range[0]) / (data_range[1] - data_range[0]) * 255.0).to(torch.uint8)).to(torch.float32)
        gts = (((data['gts'] - data_range[0]) / (data_range[1] - data_range[0]) * 255.0).to(torch.uint8)).to(torch.float32)

        return inp, gts

