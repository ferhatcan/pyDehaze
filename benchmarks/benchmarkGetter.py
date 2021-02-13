from benchmarks.benchmark_zoo import make_benchmarks

def getBenchmarks(args):
    return make_benchmarks(args)

# from utils.configParser import options
#
# CONFIG_FILE_PATH = '../configs/EncoderDecoder_v01.ini'
#
# args = options(CONFIG_FILE_PATH)
#
# benchs = getBenchmarks(args.argsBenchs)
#
# tmp = 0