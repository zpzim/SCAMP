#include <benchmark/benchmark.h>
#include "benchmark/common.h"

static void benchmarkArgsCI(benchmark::internal::Benchmark* b) {
  b->Args({-1, 1 << 20});
}

BENCHMARK(BM_1NN_INDEX_SELF_JOIN)->Apply(benchmarkArgsCI);
BENCHMARK(BM_1NN_SELF_JOIN)->Apply(benchmarkArgsCI);
BENCHMARK(BM_SUM_SELF_JOIN)->Apply(benchmarkArgsCI);
BENCHMARK(BM_MATRIX_SELF_JOIN)->Apply(benchmarkArgsCI);

BENCHMARK_MAIN();
