// Benchmark implementation for the SCAMP autotuner. Provides the
// BenchmarkFn that RunAutotuneWithBenchmark calls per (profile,
// precision, variant) trial. Lives in its own TU so it can depend on
// scamp_op (the full SCAMP API) without making gpu_utils circular: only
// callers that explicitly want the benchmarked autotune path link
// against autotune_bench, while gpu_utils stays self-contained.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "autotune.h"
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_exception.h"
#include "common/scamp_interface.h"
#include "kernel_config.h"

#ifdef _HAS_CUDA_
#include "common/cuda_to_hip.h"
#endif

namespace SCAMP {

namespace {

// Workload-shape constants. The default (256K) is sized so trial
// timings reflect steady-state kernel cost rather than launch
// overhead, while keeping a full sweep (10 targets x num_variants x
// 4 blocksz settings) tractable on a recent GPU (~25 min on an
// RTX 3080). Smaller defaults produced noisier rankings that
// occasionally flipped per-target winners between back-to-back
// sweeps; see docs/source/autotune.rst for the measured comparison.
//
// SCAMP_AUTOTUNE_INPUT_LENGTH (env) overrides this. Sweep time grows
// with N^2 (per-trial kernel cost is ~quadratic), so users on older
// or slower GPUs can dial it down (131072 = "casual" tuning that
// finishes in ~8 min) and users producing data/autotune_cache.txt
// entries to ship in a release can dial it up (524288 = ~1.5 h).
constexpr int kBenchmarkInputLengthDefault = 262144;
constexpr int kBenchmarkWindow = 200;
// Threshold matches the CLI default (--threshold=0) across all profile types.
// SUM_THRESH/MATRIX_SUMMARY/KNN all gate their write-back on this value, so
// tuning at threshold=0 measures the maximally-loaded write-back path -- the
// case the autotuner actually needs to optimize. A non-zero value (e.g. 0.5)
// filters most distances out, making the autotuner blind to the write-back
// cost and biasing it toward configs that win only on near-empty workloads.
constexpr double kBenchmarkThreshold = 0.0;
constexpr int kBenchmarkMatrixDim = 100;
// Matches the CLI / pyscamp default (--max_matches_per_column=5); the
// autotuner measures what real users actually hit.
constexpr int kBenchmarkMaxMatchesPerColumn = 5;

int BenchmarkInputLength() {
  static const int len = [] {
    const char *env = std::getenv("SCAMP_AUTOTUNE_INPUT_LENGTH");
    if (env != nullptr && env[0] != '\0') {
      try {
        int parsed = std::stoi(env);
        if (parsed > kBenchmarkWindow) return parsed;
      } catch (const std::exception &) {
      }
    }
    return kBenchmarkInputLengthDefault;
  }();
  return len;
}

std::vector<double> &SharedSyntheticInput() {
  // Reuse the same random input across all trials so per-variant timings
  // are comparable; seeding deterministically also makes the autotune
  // pass reproducible across invocations.
  static const std::vector<double> input = [] {
    int n = BenchmarkInputLength();
    std::vector<double> v(n);
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
  args->max_tile_size = 512000;
  args->distributed_start_row = -1;
  args->distributed_start_col = -1;
  args->distance_threshold = kBenchmarkThreshold;
  args->precision_type = precision;
  args->profile_type = profile;
  args->profile_a.type = profile;
  args->profile_b.type = profile;
  args->computing_rows = true;
  args->computing_columns = true;
  args->keep_rows_separate = false;
  args->is_aligned = false;
  args->silent_mode = true;
  args->max_matches_per_column = kBenchmarkMaxMatchesPerColumn;
  args->matrix_height =
      (profile == PROFILE_TYPE_MATRIX_SUMMARY) ? kBenchmarkMatrixDim : 0;
  args->matrix_width =
      (profile == PROFILE_TYPE_MATRIX_SUMMARY) ? kBenchmarkMatrixDim : 0;
}

}  // namespace

namespace {

// Single timed run of the synthetic workload with the cfg override set.
// Throws SCAMPException if the kernel launch failed (CUDA reports an
// error post-synchronize), so the variant is reported as inf time rather
// than silently winning. The post-synchronize check is load-bearing:
// some variant geometries can exceed the per-block smem limit (especially
// for SP self-join modes that need a bigger profile output region) and
// silently return SCAMP_CUDA_ERROR, which do_SCAMP swallows. Without
// the check, the benchmark would record the fast "failed" path as a win
// and the autotune would write a broken cfg to the cache.
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
// Throws SCAMPException on failure -- the caller (the
// RunAutotuneWithBenchmark loop) catches and treats failures as
// "infinitely slow."

// Per-trial warmup count. Default 0: the first kernel launch for a given
// (variant, blocksz) instantiation is typically only a few percent slower
// than steady-state because most JIT / module-load cost is amortized by
// the process-level first launch, not per-trial; the cross-target geomean
// ranking downstream tolerates a few % of noise.
//
// SCAMP_AUTOTUNE_WARMUP_RUNS (env) overrides this. Set to 1 (or more) when
// trial timings look noisy or when running on a colder GPU/driver where the
// first kernel launch of a never-before-seen template instantiation takes
// significantly longer than steady-state.
int BenchmarkWarmupRuns() {
  static const int n = [] {
    const char *env = std::getenv("SCAMP_AUTOTUNE_WARMUP_RUNS");
    if (env == nullptr || env[0] == '\0') return 0;
    try {
      int parsed = std::stoi(env);
      if (parsed >= 0) return parsed;
    } catch (const std::exception &) {
    }
    return 0;
  }();
  return n;
}

double DefaultBenchmarkVariant(int device_id, SCAMPProfileType profile,
                               SCAMPPrecisionType precision,
                               const KernelConfig &cfg) {
  constexpr int kBenchmarkTimedRuns = 1;
  const int warmups = BenchmarkWarmupRuns();
  for (int i = 0; i < warmups; ++i) {
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
