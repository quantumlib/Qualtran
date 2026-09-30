#include "kickmix/util/fixed_width_int.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(fixed_width_int_imul64) {
    std::random_device d;
    std::seed_seq seq{1, 2, 3, 4};
    std::mt19937_64 rng(seq);
    FixedWidthInt a(1024);
    uint64_t b;
    a.randomize(rng);
    b = rng();
    b |= 1;

    benchmark_go([&]() {
        a *= b;
    })
        .goal_nanos(14)
        .show_rate("words", a.num_words);

    if (!a.non_zero()) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(fixed_width_int_iadd_product) {
    std::random_device d;
    std::seed_seq seq{1, 2, 3, 4};
    std::mt19937_64 rng(seq);
    FixedWidthInt a(1024);
    FixedWidthInt b(1024);
    FixedWidthInt c(1024);
    a.randomize(rng);
    b.randomize(rng);
    c.randomize(rng);

    benchmark_go([&]() {
        c.iadd_product(a, b);
    })
        .goal_nanos(190)
        .show_rate("wordpairs", a.num_words * b.num_words);

    if (!c.non_zero()) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(fixed_width_int_iadd_product64) {
    std::random_device d;
    std::seed_seq seq{1, 2, 3, 4};
    std::mt19937_64 rng(seq);
    FixedWidthInt a(1024);
    uint64_t b;
    FixedWidthInt c(1024);
    a.randomize(rng);
    b = rng();
    c.randomize(rng);

    benchmark_go([&]() {
        c.iadd_product(a, b);
    })
        .goal_nanos(15)
        .show_rate("words", a.num_words);

    if (!c.non_zero()) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(fixed_width_int_remainder) {
    std::random_device d;
    std::seed_seq seq{1, 2, 3, 4};
    std::mt19937_64 rng(seq);
    FixedWidthInt a(1024);
    FixedWidthInt b(2048);
    FixedWidthInt c(2048);
    a.randomize(rng);
    b.randomize(rng);
    c.randomize(rng);

    benchmark_go([&]() {
        b ^= c;
        b %= a;
    })
        .goal_micros(2.4)
        .show_rate("wordpairs", a.num_words * b.num_words);

    if (!c.non_zero()) {
        std::cerr << "data dependence\n";
    }
}
