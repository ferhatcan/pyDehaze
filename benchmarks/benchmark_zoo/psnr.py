from operator import gt

import torch

from benchmarks.IBenchmark import IBenchmark

class PSNR(IBenchmark):
    def __init__(self, args):
        super(PSNR, self).__init__(args)

    def forward(self, data: dict) -> torch.tensor:
        self.checkInput(data)

        inp, gts = self.normalizeInputs(data)
        mse = torch.mean((inp - gts) ** 2)
        result = 20 * torch.log10(255.0 / torch.sqrt(mse))

        self.fillBenchmarkDict(result)
        return self.benchmarkDict

    def fillBenchmarkDict(self, result):
        self.benchmarkDict['bench_name'] = 'PSNR'
        self.benchmarkDict['compare_func'] = gt
        self.benchmarkDict['max_value'] = 100
        self.benchmarkDict['min_value'] = 0
        self.benchmarkDict['describe_text'] = 'PSNR score is {:.3f}/{:.3f}'.format(result, self.benchmarkDict['max_value'])
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

