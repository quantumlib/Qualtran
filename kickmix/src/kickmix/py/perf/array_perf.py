import kickmix as km

import benchmark_suite

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
