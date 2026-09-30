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

benchmark = benchmark_suite.BenchmarkSuite("CircuitBuilder")


@benchmark(goal_nanos=280)
def benchmark_ccx_qqq_1():
    builder = km.CircuitBuilder()
    a, b, c = [km.q(k) for k in range(3)]

    def run():
        builder.ccx(a, b, c)

    return benchmark.go(run, rates={'ops': 1})


@benchmark(goal_nanos=950)
def benchmark_ccx_qqq_256():
    n = 256
    builder = km.CircuitBuilder()
    c1 = km.array([km.q(k) for k in range(0, n)])
    c2 = km.array([km.q(k) for k in range(n, 2 * n)])
    t = km.array([km.q(k) for k in range(2 * n, 3 * n)])

    def run():
        builder.ccx(c1, c2, t)

    return benchmark.go(run, rates={'ops': n})


@benchmark(goal_nanos=700)
def benchmark_ccx_cqq_256():
    n = 256
    builder = km.CircuitBuilder()
    c1 = km.array([k % 3 > 0 for k in range(0, n)])
    c2 = km.array([km.q(k) for k in range(n, 2 * n)])
    t = km.array([km.q(k) for k in range(2 * n, 3 * n)])

    def run():
        builder.ccx(c1, c2, t)

    return benchmark.go(run, rates={'ops': n})


@benchmark(goal_nanos=3300)
def benchmark_ccx_mixed_256():
    n = 256
    builder = km.CircuitBuilder()
    c1 = km.array([True if k % 2 == 0 else km.q(k) for k in range(0, n)])
    c2 = km.array([km.q(k) if k % 3 == 0 else km.b(k) for k in range(n, 2 * n)])
    t = km.array([km.q(k) for k in range(2 * n, 3 * n)])

    def run():
        builder.ccx(c1, c2, t)

    return benchmark.go(run, rates={'ops': n})


@benchmark(goal_nanos=3300)
def benchmark_iadd_256():
    n = 256
    builder = km.CircuitBuilder()
    c1 = builder.create_quantum_register(n, "offset")
    c2 = builder.create_quantum_register(n, "target")

    def run():
        builder.iadd(c2, target=c1)

    return benchmark.go(run, rates={'qubits': n})


if __name__ == '__main__':
    benchmark.main()
