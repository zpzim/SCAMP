#include <random>

#include <benchmark/benchmark.h>
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_exception.h"
#include "common/scamp_utils.h"
#include "common/scamp_interface.h"


double get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0,1);
  return dis(e);
}

std::vector<double> get_random_vec(size_t size) {
  std::vector<double> out(size);
  for (int i = 0; i < size; ++i) {
    out[i] = get_random();
  }
  return out;
}

static void benchmarkArgsCI(benchmark::internal::Benchmark* b) {
  for (int i = 1; i < 3; ++i) {
    for (int j = 15; j < 18; ++j) {
      b->Args({i, 1 << j});
    }
  }
}

static void BM_1NN_INDEX_SELF_JOIN(benchmark::State& state) {

  // 128K random vector
  std::vector<double> ts = get_random_vec(state.range(1));  
  
  std::uniform_real_distribution<> dis(0, 1);

  SCAMP::SCAMPArgs args;
  args.window = 100;
  args.max_tile_size = 1 << 17;
  args.has_b = false;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = -1;
  args.computing_columns = true;
  args.computing_rows = true;
  args.profile_a.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  args.profile_b.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.timeseries_a = ts;
  args.timeseries_b = ts;
  args.silent_mode = true;
  args.max_matches_per_column = 1;
  args.matrix_height = 0;
  args.matrix_width = 0;

  std::vector<int> gpu_devices;
  int num_threads = state.range(0);

  for (auto _ : state) {
    SCAMP::do_SCAMP(&args, gpu_devices, num_threads);
  }

}

BENCHMARK(BM_1NN_INDEX_SELF_JOIN)->Apply(benchmarkArgsCI);

static void BM_1NN_SELF_JOIN(benchmark::State& state) {

  // 128K random vector
  std::vector<double> ts = get_random_vec(state.range(1));  
  
  std::uniform_real_distribution<> dis(0, 1);

  SCAMP::SCAMPArgs args;
  args.window = 100;
  args.max_tile_size = 1 << 17;
  args.has_b = false;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = -1;
  args.computing_columns = true;
  args.computing_rows = true;
  args.profile_a.type = SCAMP::PROFILE_TYPE_1NN;
  args.profile_b.type = SCAMP::PROFILE_TYPE_1NN;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = SCAMP::PROFILE_TYPE_1NN;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.timeseries_a = ts;
  args.timeseries_b = ts;
  args.silent_mode = true;
  args.max_matches_per_column = 1;
  args.matrix_height = 0;
  args.matrix_width = 0;

  std::vector<int> gpu_devices;
  int num_threads = state.range(0);

  for (auto _ : state) {
    SCAMP::do_SCAMP(&args, gpu_devices, num_threads);
  }
}

BENCHMARK(BM_1NN_SELF_JOIN)->Apply(benchmarkArgsCI);

static void BM_SUM_SELF_JOIN(benchmark::State& state) {

  // 128K random vector
  std::vector<double> ts = get_random_vec(state.range(1));  
  
  std::uniform_real_distribution<> dis(0, 1);

  SCAMP::SCAMPArgs args;
  args.window = 100;
  args.max_tile_size = 1 << 17;
  args.has_b = false;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = -1;
  args.computing_columns = true;
  args.computing_rows = true;
  args.profile_a.type = SCAMP::PROFILE_TYPE_SUM_THRESH;
  args.profile_b.type = SCAMP::PROFILE_TYPE_SUM_THRESH;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = SCAMP::PROFILE_TYPE_SUM_THRESH;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.timeseries_a = ts;
  args.timeseries_b = ts;
  args.silent_mode = true;
  args.max_matches_per_column = 1;
  args.matrix_height = 0;
  args.matrix_width = 0;

  std::vector<int> gpu_devices;
  int num_threads = state.range(0);

  for (auto _ : state) {
    SCAMP::do_SCAMP(&args, gpu_devices, num_threads);
  }
}

BENCHMARK(BM_SUM_SELF_JOIN)->Apply(benchmarkArgsCI);

BENCHMARK_MAIN();
