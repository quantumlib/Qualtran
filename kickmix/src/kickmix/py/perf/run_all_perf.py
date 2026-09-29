import glob
import importlib
import importlib.util
import pathlib

import benchmark_suite


def main():
    for path in glob.glob(str(pathlib.Path(__file__).parent) + '/**_perf.py', recursive=True):
        spec = importlib.util.spec_from_file_location('<anon>', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'benchmark'):
            if isinstance(module.benchmark, benchmark_suite.BenchmarkSuite):
                module.benchmark.main()


if __name__ == '__main__':
    main()
