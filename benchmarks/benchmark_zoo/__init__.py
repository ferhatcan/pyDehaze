from .mse import MSE
from .psnr import PSNR

__all__ = ['MSE', 'PSNR']

def make_benchmarks(args):
    benchmarks = dict()
    benchmarks['bench_methods'] = []
    benchmarks['name'] = []

    for bench in args.benchmarks.split(','):
        bench = bench.strip()
        if bench.upper() in __all__:
            benchmarks['name'].append(bench.upper())
            benchmarks['bench_methods'].append(globals()[bench.upper()](args))
        else:
            print('There is no benchmark method exist with name {}'.format(bench.upper()))

    return benchmarks