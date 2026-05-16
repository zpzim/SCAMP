// Public autotune API.
//
// Lookup at kernel launch time (see GetKernelConfigForDevice) consults three
// sources in order, falling through whenever the higher-priority source has
// no entry for the (device, profile_type, precision) tuple:
//
//   1. User override file. Defaults to $SCAMP_AUTOTUNE_CACHE or
//      ~/.cache/scamp/autotune.txt. RunAutotune() writes here.
//   2. Built-in cache embedded in the binary at build time from
//      data/autotune_cache.txt (see builtin_autotune_cache.h). This is what
//      conda-forge / pip-wheel users actually benefit from, since they
//      cannot recompile to add kernel variants.
//   3. GetDefaultKernelConfig() -- the compile-time default.
//
// Workflow for shipping a new device's tuned config:
//   1. Build SCAMP / pyscamp from source on that device.
//   2. Run `SCAMP --autotune` (or pyscamp.autotune()). The override file is
//      populated.
//   3. Merge the new device's lines into data/autotune_cache.txt and open
//      a PR. Future releases will ship those entries to end users.
//
// Today only the compile-time default KernelConfig is honored by the
// dispatcher (IsSupportedKernelConfig returns true only for that tuple);
// cache entries with different values are silently fallback-skipped. The
// infrastructure is in place so that follow-up PRs which add real kernel
// variants will make the cache load-bearing for end users.
#pragma once

#include <string>
#include "common/common.h"
#include "kernel_config.h"

namespace SCAMP {

struct AutotuneResult {
  std::string device_name;  // Sanitized device key as it appears in the cache.
  std::string cache_path;   // Where the cache was loaded/saved.
  KernelConfig chosen;      // Config the autotuner selected (default for now).
  bool wrote_cache;         // True if the cache file was rewritten.
};

// Run the autotune workflow for one device:
//   1. Query device properties
//   2. For each (profile_type, precision) pair, pick a KernelConfig (today:
//      GetDefaultKernelConfig). When kernel variants exist, this is where
//      benchmark+select will happen.
//   3. Persist results to the cache file.
//
// Returns a summary of what was chosen. Throws SCAMPException on CUDA or
// filesystem failures.
//
// cache_path is optional; pass empty string to use
// AutotuneCache::DefaultPath().
AutotuneResult RunAutotune(int device_id, const std::string &cache_path = "",
                           bool verbose = true);

// Look up the configuration the autotuner persisted for this device, falling
// back to GetDefaultKernelConfig(precision) if the cache has no entry, the
// cache file is missing, or the cached entry is not a kernel variant the
// current binary was compiled with. Always returns a usable config.
//
// This is safe to call from the kernel launch hot path: the cache is loaded
// lazily on first call and reused thereafter. Pass cache_path="" to use the
// default location.
KernelConfig GetKernelConfigForDevice(int device_id,
                                      SCAMPProfileType profile_type,
                                      SCAMPPrecisionType precision,
                                      const std::string &cache_path = "");

}  // namespace SCAMP
