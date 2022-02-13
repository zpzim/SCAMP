#include <benchmark/benchmark.h>

void BM_1NN_INDEX_SELF_JOIN(benchmark::State& state);
void BM_1NN_SELF_JOIN(benchmark::State& state);
void BM_SUM_SELF_JOIN(benchmark::State& state);
void BM_MATRIX_SELF_JOIN(benchmark::State& state);
