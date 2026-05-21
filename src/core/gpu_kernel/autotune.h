// Public autotune API.
//
// Lookup at kernel launch time (see GetKernelConfigForDevice) consults
// these sources in order:
//
//   0. Thread-local override (Set/ClearKernelConfigOverride). Used by
//      the autotune benchmark loop to force one specific variant per
//      timed run; not normally set.
//   1. User override file. Defaults to $SCAMP_AUTOTUNE_CACHE or
//      ~/.cache/scamp/autotune.txt. RunAutotuneWithBenchmark writes here.
//   2. Built-in cache embedded in the binary at build time from
//      data/autotune_cache.txt (see builtin_autotune_cache.h). Ships
//      with the binary so conda-forge / pip-wheel users get tuned
//      configs without needing to recompile.
//   3. GetDefaultKernelConfig() -- the compile-time default
//      (kKernelVariants[0]).
//
// Workflow for shipping a new device's tuned config:
//   1. Build SCAMP / pyscamp from source on that device.
//   2. Run `SCAMP --autotune` (or pyscamp.autotune()) which calls
//      RunAutotuneWithBenchmark. The override file is populated with
//      the per-(profile, precision) variant that benchmarked fastest.
//   3. Merge the new device's lines into data/autotune_cache.txt and
//      open a PR. Future releases will ship those entries to end users.
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

// Backward-compat thin wrapper: writes the compile-time default
// (kKernelVariants[0]) for every (profile, precision) tuple. Does NOT
// benchmark variants. Use RunAutotuneWithBenchmark to actually tune.
AutotuneResult RunAutotune(int device_id, const std::string &cache_path = "",
                           bool verbose = true);

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
// tests can exercise the user-cache -> built-in -> fallback chain + the
// cache-miss warning without needing a real CUDA device. Production
// callers should use GetKernelConfigForDevice instead.
//
// `device_key`     -- the sanitized device key (as it appears in the
//                     cache file).
// `user_cache`     -- the user override cache (may be nullptr to skip).
// `builtin_cache`  -- the built-in cache embedded at build time (may
//                     be nullptr to skip).
// `fallback`       -- compile-time default returned on miss.
//
// On miss (neither cache has a usable entry), emits a one-shot warning
// to stderr identifying the missing (device, profile, precision) tuple
// and what config got used as a fallback. Subsequent misses for the
// same tuple are silent. ResetAutotuneWarnings() clears the dedup set
// (used by tests).
class AutotuneCache;  // fwd
KernelConfig LookupKernelConfigForDeviceKey(const std::string &device_key,
                                            SCAMPProfileType profile_type,
                                            SCAMPPrecisionType precision,
                                            const AutotuneCache *user_cache,
                                            const AutotuneCache *builtin_cache,
                                            const KernelConfig &fallback);

// Clear the cache-miss warning dedup set. Tests use this; production
// code shouldn't need to call it.
void ResetAutotuneWarnings();

// Set / clear a thread-local KernelConfig override. While set,
// GetKernelConfigForDevice returns the override and skips the cache
// lookup. The autotune benchmark loop uses this to force one specific
// variant per timed run.
//
// Override must be cleared before the thread returns to normal use,
// otherwise the override leaks into subsequent SCAMP calls on the same
// thread. (Pattern: RAII scope guard.)
void SetKernelConfigOverride(const KernelConfig &cfg);
void ClearKernelConfigOverride();

}  // namespace SCAMP
