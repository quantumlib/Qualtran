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

import benchmark_suite

import kickmix as km

benchmark = benchmark_suite.BenchmarkSuite("km.array")


@benchmark(goal_micros=2.8)
def benchmark_construct():
    items = []
    for k in range(100):
        items.append(km.q(k))
    for k in range(100):
        items.append(km.b(k))
    for k in range(100):
        items.append(k % 2 == 0)

    def run():
        km.array(items)

    return benchmark.go(run, rates={'items': len(items)})


if __name__ == '__main__':
    benchmark.main()
