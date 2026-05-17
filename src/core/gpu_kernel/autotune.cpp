#include "autotune.h"

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "autotune_cache.h"
#include "builtin_autotune_cache.h"
#include "common/scamp_exception.h"
#include "device_props.h"
#include "kernel_config.h"

namespace SCAMP {

namespace {

// The set of (profile_type, precision) pairs the autotuner emits records
// for. Kept compact: we don't include profile types the GPU path doesn't
// support today (FREQUENCY_THRESH, KNN, 1NN_MULTIDIM).
struct ProfilePrecisionPair {
  SCAMPProfileType profile;
  SCAMPPrecisionType precision;
};

constexpr std::array<ProfilePrecisionPair, 15> kAutotuneTargets{{
    {PROFILE_TYPE_1NN_INDEX, PRECISION_DOUBLE},
    {PROFILE_TYPE_1NN_INDEX, PRECISION_MIXED},
    {PROFILE_TYPE_1NN_INDEX, PRECISION_SINGLE},
    {PROFILE_TYPE_1NN, PRECISION_DOUBLE},
    {PROFILE_TYPE_1NN, PRECISION_MIXED},
    {PROFILE_TYPE_1NN, PRECISION_SINGLE},
    {PROFILE_TYPE_SUM_THRESH, PRECISION_DOUBLE},
    {PROFILE_TYPE_SUM_THRESH, PRECISION_MIXED},
    {PROFILE_TYPE_SUM_THRESH, PRECISION_SINGLE},
    {PROFILE_TYPE_MATRIX_SUMMARY, PRECISION_DOUBLE},
    {PROFILE_TYPE_MATRIX_SUMMARY, PRECISION_MIXED},
    {PROFILE_TYPE_MATRIX_SUMMARY, PRECISION_SINGLE},
    {PROFILE_TYPE_APPROX_ALL_NEIGHBORS, PRECISION_DOUBLE},
    {PROFILE_TYPE_APPROX_ALL_NEIGHBORS, PRECISION_MIXED},
    {PROFILE_TYPE_APPROX_ALL_NEIGHBORS, PRECISION_SINGLE},
}};

// Process-wide lazy caches: loaded once on first lookup, cleared if a
// different cache_path is requested.
//   user_cache    -- file at $SCAMP_AUTOTUNE_CACHE / ~/.cache/scamp/...
//                    Devs write here when running RunAutotune locally.
//   builtin_cache -- parsed from kBuiltinAutotuneCache (embedded at build
//                    time from data/autotune_cache.txt). Ships with the
//                    binary; conda-forge / pip-wheel users rely on this.
//
// We do not currently invalidate either cache when the underlying source
// changes -- callers that want fresh data should restart the process or
// call RunAutotune() which loads + saves explicitly.
struct CacheState {
  std::unique_ptr<AutotuneCache> user_cache;
  std::string loaded_user_path;
  std::unique_ptr<AutotuneCache> builtin_cache;
};

CacheState &SharedCacheState() {
  static CacheState state;
  return state;
}

std::mutex &SharedCacheMutex() {
  static std::mutex m;
  return m;
}

const AutotuneCache *GetOrLoadUserCache(const std::string &cache_path) {
  // Caller must hold SharedCacheMutex().
  CacheState &state = SharedCacheState();
  std::string resolved =
      cache_path.empty() ? AutotuneCache::DefaultPath() : cache_path;
  if (state.user_cache && state.loaded_user_path == resolved) {
    return state.user_cache.get();
  }
  state.user_cache = std::make_unique<AutotuneCache>(resolved);
  state.loaded_user_path = resolved;
  state.user_cache->Load();
  return state.user_cache.get();
}

const AutotuneCache *GetOrLoadBuiltinCache() {
  // Caller must hold SharedCacheMutex().
  CacheState &state = SharedCacheState();
  if (state.builtin_cache) return state.builtin_cache.get();
  state.builtin_cache = std::make_unique<AutotuneCache>("<builtin>");
  state.builtin_cache->LoadFromString(kBuiltinAutotuneCache);
  return state.builtin_cache.get();
}

}  // namespace

AutotuneResult RunAutotune(int device_id, const std::string &cache_path,
                           bool verbose) {
  GpuDeviceProps props = QueryDeviceProps(device_id);
  std::string device_key = props.CacheKey();
  std::string resolved =
      cache_path.empty() ? AutotuneCache::DefaultPath() : cache_path;

  AutotuneCache disk_cache(resolved);
  // Load any existing entries so we don't clobber records from other devices
  // that share the same cache file.
  disk_cache.Load();

  if (verbose) {
    std::cout << "SCAMP autotune\n"
              << "  device      : " << props.name << " (sm_"
              << props.compute_major << props.compute_minor << ", "
              << props.sm_count << " SMs)\n"
              << "  override    : " << resolved << "\n"
              << "  device key  : " << device_key << "\n"
              << "\n"
              << "  NOTE: Results are written to the user override path\n"
              << "  above. To ship these defaults to end users (incl.\n"
              << "  conda-forge / pip-wheel installs that cannot recompile),\n"
              << "  merge the new lines into data/autotune_cache.txt and\n"
              << "  open a PR. The built-in cache embedded in the binary is\n"
              << "  refreshed from that file at build time.\n"
              << "\n";
  }

  KernelConfig last_chosen{};
  for (const auto &t : kAutotuneTargets) {
    // Today there is only one supported KernelConfig per (profile, precision):
    // the compile-time default. Future PRs will add alternative variants and
    // benchmark them here, then pick the fastest. The cache file is already
    // shaped to receive per-tuple records.
    KernelConfig cfg = GetDefaultKernelConfig(t.precision);
    disk_cache.Store(device_key, t.profile, t.precision, cfg);
    last_chosen = cfg;
    if (verbose) {
      std::cout << "  " << ProfileTypeName(t.profile) << " "
                << PrecisionTypeName(t.precision) << " -> blocksz=" << cfg.blocksz
                << " bps=" << cfg.blocks_per_sm
                << " dpt=" << cfg.diags_per_thread
                << " ur=" << cfg.unrolled_rows
                << " our=" << cfg.outer_unrolled_rows
                << " kti=" << cfg.kernel_tile_iters
                << " (tile_height=" << cfg.tile_height() << ")\n";
    }
  }

  disk_cache.Save();

  // Invalidate any process-wide cached copy of the user override so
  // subsequent kernel launches pick up the fresh entries.
  {
    std::lock_guard<std::mutex> lock(SharedCacheMutex());
    SharedCacheState().user_cache.reset();
    SharedCacheState().loaded_user_path.clear();
  }

  if (verbose) {
    std::cout << "  wrote " << kAutotuneTargets.size() << " entries to "
              << resolved << std::endl;
  }

  return AutotuneResult{device_key, resolved, last_chosen, true};
}

KernelConfig GetKernelConfigForDevice(int device_id,
                                      SCAMPProfileType profile_type,
                                      SCAMPPrecisionType precision,
                                      const std::string &cache_path) {
  KernelConfig fallback = GetDefaultKernelConfig(precision);
  try {
    GpuDeviceProps props = QueryDeviceProps(device_id);
    std::string key = props.CacheKey();

    std::lock_guard<std::mutex> lock(SharedCacheMutex());

    // 1) User override (env var or ~/.cache/scamp/autotune.txt).
    const AutotuneCache *user = GetOrLoadUserCache(cache_path);
    auto user_hit = user->Lookup(key, profile_type, precision);
    if (user_hit.has_value() &&
        IsSupportedKernelConfig(*user_hit, precision)) {
      return *user_hit;
    }

    // 2) Built-in cache embedded at build time from data/autotune_cache.txt.
    const AutotuneCache *builtin = GetOrLoadBuiltinCache();
    auto builtin_hit = builtin->Lookup(key, profile_type, precision);
    if (builtin_hit.has_value() &&
        IsSupportedKernelConfig(*builtin_hit, precision)) {
      return *builtin_hit;
    }

    // 3) Compile-time default.
    return fallback;
  } catch (const SCAMPException &) {
    return fallback;
  } catch (const std::exception &) {
    return fallback;
  }
}

}  // namespace SCAMP
