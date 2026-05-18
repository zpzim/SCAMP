// Benchmark implementation for the SCAMP autotuner. Provides the
// BenchmarkFn that RunAutotuneWithBenchmark calls per (profile,
// precision, variant) trial. Lives in its own TU so it can depend on
// scamp_op (the full SCAMP API) without making gpu_utils circular: only
// callers that explicitly want the benchmarked autotune path link
// against autotune_bench, while gpu_utils stays self-contained.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "autotune.h"
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_exception.h"
#include "common/scamp_interface.h"
#include "kernel_config.h"

#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#endif

namespace SCAMP {

namespace {

// Workload-shape constants. Small enough to keep a full autotune sweep
// (kAutotuneTargets x kNumKernelVariants trials) under a minute on a
// fast GPU, large enough to amortize launch overhead so the timing
// reflects steady-state kernel cost.
constexpr int kBenchmarkInputLength = 65536;
constexpr int kBenchmarkWindow = 200;
constexpr double kBenchmarkSumThreshold = 0.5;
constexpr int kBenchmarkMatrixDim = 100;
constexpr int kBenchmarkMaxMatchesPerColumn = 5;

std::vector<double> &SharedSyntheticInput() {
  // Reuse the same random input across all trials so per-variant timings
  // are comparable; seeding deterministically also makes the autotune
  // pass reproducible across invocations.
  static const std::vector<double> input = [] {
    std::vector<double> v(kBenchmarkInputLength);
    std::mt19937_64 rng(0xDEADBEEFULL);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &x : v) x = dist(rng);
    return v;
  }();
  return const_cast<std::vector<double> &>(input);
}

void PopulateBenchmarkArgs(SCAMPArgs *args, SCAMPProfileType profile,
                           SCAMPPrecisionType precision) {
  args->timeseries_a = SharedSyntheticInput();
  args->timeseries_b.clear();
  args->has_b = false;
  args->window = kBenchmarkWindow;
  args->max_tile_size = 131072;
  args->distributed_start_row = -1;
  args->distributed_start_col = -1;
  args->distance_threshold = kBenchmarkSumThreshold;
  args->precision_type = precision;
  args->profile_type = profile;
  args->profile_a.type = profile;
  args->profile_b.type = profile;
  args->computing_rows = false;
  args->computing_columns = true;
  args->keep_rows_separate = false;
  args->is_aligned = false;
  args->silent_mode = true;
  args->max_matches_per_column = kBenchmarkMaxMatchesPerColumn;
  args->matrix_height = (profile == PROFILE_TYPE_MATRIX_SUMMARY)
                            ? kBenchmarkMatrixDim
                            : 0;
  args->matrix_width = (profile == PROFILE_TYPE_MATRIX_SUMMARY)
                           ? kBenchmarkMatrixDim
                           : 0;
}

}  // namespace

namespace {

// Single timed run of the synthetic workload with the cfg override set.
// Throws SCAMPException if the kernel launch failed (CUDA reports an
// error post-synchronize), so the variant is reported as inf time rather
// than silently winning. This caught a real bug: v5 (OUR=16) exceeds the
// default 48KB per-block smem limit for SP self-join modes and silently
// returns SCAMP_CUDA_ERROR, which do_SCAMP swallows. Without the
// post-synchronize check the benchmark recorded the fast "failed" path
// as the winning time and the autotune wrote a broken cfg to the cache.
double TimeOneRun(int device_id, SCAMPProfileType profile,
                  SCAMPPrecisionType precision, const KernelConfig &cfg) {
  SCAMPArgs args;
  PopulateBenchmarkArgs(&args, profile, precision);

  SetKernelConfigOverride(cfg);
  struct OverrideGuard {
    ~OverrideGuard() { ClearKernelConfigOverride(); }
  } guard;

  std::vector<int> devices{device_id};

#ifdef _HAS_CUDA_
  // Clear any sticky CUDA error from a prior trial so we only see this
  // trial's launches.
  cudaGetLastError();
#endif
  auto start = std::chrono::steady_clock::now();
  do_SCAMP(&args, devices, /*num_threads=*/0);
#ifdef _HAS_CUDA_
  cudaError_t sync_err = cudaDeviceSynchronize();
  cudaError_t async_err = cudaGetLastError();
  if (sync_err != cudaSuccess || async_err != cudaSuccess) {
    cudaError_t err = sync_err != cudaSuccess ? sync_err : async_err;
    throw SCAMPException(std::string("benchmark variant CUDA error: ") +
                         cudaGetErrorString(err));
  }
#endif
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

}  // namespace

// Default BenchmarkFn: builds a synthetic self-join workload for the
// given (profile, precision), times kBenchmarkTimedRuns runs of it with
// the cfg override set, and returns the MIN seconds across runs.
//
// Why min-of-N rather than mean/median: kernel runtime is a hard lower
// bound (no run can go faster than the actual compute cost); noise only
// makes runs slower (preemption, thermal throttle, contention). The
// minimum is the most faithful estimate of the steady-state cost.
//
// kBenchmarkWarmupRuns runs are discarded first to amortize one-time
// costs (CUDA module load, JIT, cache fill).
//
// Throws SCAMPException on failure -- the caller (the
// RunAutotuneWithBenchmark loop) catches and treats failures as
// "infinitely slow."
double DefaultBenchmarkVariant(int device_id, SCAMPProfileType profile,
                               SCAMPPrecisionType precision,
                               const KernelConfig &cfg) {
  constexpr int kBenchmarkWarmupRuns = 1;
  constexpr int kBenchmarkTimedRuns = 3;
  for (int i = 0; i < kBenchmarkWarmupRuns; ++i) {
    (void)TimeOneRun(device_id, profile, precision, cfg);
  }
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < kBenchmarkTimedRuns; ++i) {
    double t = TimeOneRun(device_id, profile, precision, cfg);
    if (t < best) best = t;
  }
  return best;
}

}  // namespace SCAMP
