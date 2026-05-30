// Public autotune API.
//
// Lookup at kernel launch time (see GetKernelConfigForDevice) consults
// these sources in order:
//
//   0. Process-wide override (Set/ClearKernelConfigOverride). Used by
//      the autotune benchmark loop to force one specific variant per
//      timed run; not normally set.
//   1. User override file. Path comes from AutotuneCache::DefaultPath()
//      ($SCAMP_AUTOTUNE_CACHE, then $XDG_CACHE_HOME/scamp/autotune.txt,
//      then $HOME/.cache/scamp/autotune.txt on Linux/macOS or
//      %LOCALAPPDATA%\scamp\autotune.txt on Windows).
//      RunAutotuneWithBenchmark writes here.
//   2. GetDefaultKernelConfig(profile_type, precision) -- compile-time
//      default. Per-profile shfl vs sliding-window family preference;
//      the chosen variant's DefaultBlockszForPrecision picks the
//      precision-specific blocksz.
//
// Workflow for getting a tuned config:
//   1. Build SCAMP / pyscamp from source on the target device.
//   2. Run `SCAMP --autotune` (or pyscamp.autotune()) once -- it
//      writes per-(profile, precision) winners to the user-cache
//      path above, and subsequent runs read from there automatically.
#pragma once

#include <functional>
#include <string>
#include "common/common.h"
#include "kernel_config.h"

namespace SCAMP {

struct AutotuneResult {
  std::string device_name;  // Sanitized device key as it appears in the cache.
  std::string cache_path;   // Where the cache was loaded/saved.
  KernelConfig chosen;      // Config the autotuner selected for the last
                            // (profile, precision) tuple processed (mostly
                            // useful as a sanity check).
  bool wrote_cache;         // True if the cache file was rewritten.
};

// Signature for the per-trial workload runner. Implementations live in
// autotune_bench.cpp (which depends on scamp_op); the function is
// declared here so callers of RunAutotuneWithBenchmark can pass any
// custom implementation if needed (e.g., a different synthetic-input
// size for testing).
//
// Receives: device_id, the profile type and precision being tuned, and
// the KernelConfig variant to time. Implementation must call
// SetKernelConfigOverride(cfg) around its SCAMP invocation so the
// kernel actually uses the variant under test. Returns wall-clock
// seconds; SCAMPException-on-failure is OK and treated as "infinite"
// time (i.e., this variant loses).
using BenchmarkFn = std::function<double(
    int device_id, SCAMPProfileType profile_type, SCAMPPrecisionType precision,
    const KernelConfig &cfg)>;

// Full autotune: for each (profile, precision) tuple, time every variant
// in kKernelVariants via `bench` and persist the winner to the cache.
//
// Thrown SCAMPException propagates (CUDA query failed, cache write
// failed, etc.); per-variant benchmark failures are absorbed and the
// variant is treated as infinitely slow.
AutotuneResult RunAutotuneWithBenchmark(int device_id, BenchmarkFn bench,
                                        const std::string &cache_path = "",
                                        bool verbose = true);

// Lower-level overload: identical to RunAutotuneWithBenchmark but takes
// the resolved device_key directly instead of querying CUDA for it.
//
// device_id is passed through to `bench` (the bench impl in
// autotune_bench.cpp uses it; a synthetic test bench can ignore it).
// print_banner=false suppresses the device-key banner (the wrapper sets
// this when it already printed a richer banner of its own).
//
// Exists so unit tests can exercise the full bench-driven autotune
// pathway -- variant sweep, winner selection, disk write, and re-load
// -- on hosts without a CUDA device. Production code should call
// RunAutotuneWithBenchmark(device_id, ...) which builds the device_key
// from a real cudaDeviceProp.
AutotuneResult RunAutotuneWithBenchmarkForDeviceKey(
    const std::string &device_key, int device_id, BenchmarkFn bench,
    const std::string &cache_path = "", bool verbose = true,
    bool print_banner = true);

// Look up the configuration the autotuner persisted for this device,
// falling back to GetDefaultKernelConfig(precision) if no source has a
// match (or the entry isn't a kernel variant the current binary was
// compiled with). Always returns a usable config.
//
// Safe to call from the kernel launch hot path: caches are loaded
// lazily on first call. Pass cache_path="" to use the default location.
KernelConfig GetKernelConfigForDevice(int device_id,
                                      SCAMPProfileType profile_type,
                                      SCAMPPrecisionType precision,
                                      const std::string &cache_path = "");

// Pure-logic lookup used by GetKernelConfigForDevice. Exposed so unit
// tests can exercise the user-cache -> fallback chain + the cache-miss
// without needing a real CUDA device. Production callers should use
// GetKernelConfigForDevice instead.
//
// `device_key`     -- the sanitized device key (as it appears in the
//                     cache file).
// `user_cache`     -- the user override cache (may be nullptr to skip).
// `fallback`       -- compile-time default returned on miss.
class AutotuneCache;  // fwd
KernelConfig LookupKernelConfigForDeviceKey(const std::string &device_key,
                                            SCAMPProfileType profile_type,
                                            SCAMPPrecisionType precision,
                                            const AutotuneCache *user_cache,
                                            const KernelConfig &fallback);

// Set / clear a process-wide KernelConfig override. While set,
// GetKernelConfigForDevice returns the override and skips the cache
// lookup. The autotune benchmark loop uses this to force one specific
// variant per timed run; it has to be process-wide because do_SCAMP
// dispatches work to std::async workers that would not see a
// thread_local set on the autotune main thread.
//
// Override must be cleared before returning to normal use, otherwise
// it leaks into subsequent SCAMP calls in the process. (Pattern: RAII
// scope guard.)
void SetKernelConfigOverride(const KernelConfig &cfg);
void ClearKernelConfigOverride();

}  // namespace SCAMP
