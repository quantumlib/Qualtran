#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import glob
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
