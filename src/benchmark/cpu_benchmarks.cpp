#include <benchmark/benchmark.h>
#include "benchmark/common.h"

static void benchmarkArgsCI(benchmark::internal::Benchmark* b) {
  b->Args({1, 1 << 15});
}

BENCHMARK(BM_1NN_INDEX_SELF_JOIN)
    ->Unit(benchmark::kSecond)
    ->Apply(benchmarkArgsCI);
BENCHMARK(BM_1NN_SELF_JOIN)->Unit(benchmark::kSecond)->Apply(benchmarkArgsCI);
BENCHMARK(BM_SUM_SELF_JOIN)->Unit(benchmark::kSecond)->Apply(benchmarkArgsCI);
BENCHMARK(BM_MATRIX_SELF_JOIN)
    ->Unit(benchmark::kSecond)
    ->Apply(benchmarkArgsCI);

BENCHMARK_MAIN();
