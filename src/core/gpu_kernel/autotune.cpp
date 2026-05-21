#include "autotune.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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

constexpr std::array<ProfilePrecisionPair, 10> kAutotuneTargets{{
    {PROFILE_TYPE_1NN_INDEX, PRECISION_DOUBLE},
    {PROFILE_TYPE_1NN_INDEX, PRECISION_SINGLE},
    {PROFILE_TYPE_1NN, PRECISION_DOUBLE},
    {PROFILE_TYPE_1NN, PRECISION_SINGLE},
    {PROFILE_TYPE_SUM_THRESH, PRECISION_DOUBLE},
    {PROFILE_TYPE_SUM_THRESH, PRECISION_SINGLE},
    {PROFILE_TYPE_MATRIX_SUMMARY, PRECISION_DOUBLE},
    {PROFILE_TYPE_MATRIX_SUMMARY, PRECISION_SINGLE},
    {PROFILE_TYPE_APPROX_ALL_NEIGHBORS, PRECISION_DOUBLE},
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

// Process-wide override used by the autotune benchmark loop to force a
// specific KernelConfig for one timed run. While set, GetKernelConfigForDevice
// returns it and skips the cache + supportedness checks (the caller is
// responsible for only setting cfgs that map to a real variant).
//
// MUST be process-wide (not thread_local): do_SCAMP spawns worker threads via
// std::async, and the worker threads are the ones that actually call
// GetKernelConfigForDevice via compute_gpu_resources_and_launch. A
// thread_local override set on the autotune's main thread would never reach
// the workers, so they'd silently fall back to the on-disk cache and every
// trial in the sweep would launch the SAME variant -- making the
// per-variant timings indistinguishable.
//
// Synchronization: the autotune benchmark loop is sequential
// (RunAutotuneWithBenchmark runs one TimeOneRun at a time), so the
// override is set, the trial runs to completion (including
// cudaDeviceSynchronize), and then the override is cleared before the
// next trial. A mutex on the override variable guards against UB if a
// future caller decides to parallelize trials, but the steady-state path
// is uncontended.
std::mutex g_cfg_override_mutex;
std::optional<KernelConfig> g_cfg_override;

std::optional<KernelConfig> GetKernelConfigOverride() {
  std::lock_guard<std::mutex> lock(g_cfg_override_mutex);
  return g_cfg_override;
}

}  // namespace

void SetKernelConfigOverride(const KernelConfig &cfg) {
  std::lock_guard<std::mutex> lock(g_cfg_override_mutex);
  g_cfg_override = cfg;
}

void ClearKernelConfigOverride() {
  std::lock_guard<std::mutex> lock(g_cfg_override_mutex);
  g_cfg_override.reset();
}

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
                << PrecisionTypeName(t.precision)
                << " -> blocksz=" << cfg.blocksz << " bps=" << cfg.blocks_per_sm
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

AutotuneResult RunAutotuneWithBenchmark(int device_id, BenchmarkFn bench,
                                        const std::string &cache_path,
                                        bool verbose) {
  GpuDeviceProps props = QueryDeviceProps(device_id);
  std::string device_key = props.CacheKey();
  std::string resolved =
      cache_path.empty() ? AutotuneCache::DefaultPath() : cache_path;

  AutotuneCache disk_cache(resolved);
  // Load existing entries so we don't clobber records from other devices that
  // share the cache file.
  disk_cache.Load();

  if (verbose) {
    std::cout << "SCAMP autotune (benchmarked)\n"
              << "  device      : " << props.name << " (sm_"
              << props.compute_major << props.compute_minor << ", "
              << props.sm_count << " SMs)\n"
              << "  override    : " << resolved << "\n"
              << "  device key  : " << device_key << "\n"
              << "  variants    : " << kNumKernelVariants << "\n"
              << "  trials/tuple: " << kNumKernelVariants
              << " (one per variant)\n"
              << "\n"
              << "  NOTE: Per-(profile, precision) winners are written to the\n"
              << "  user override path. To ship them, merge the relevant\n"
              << "  lines into data/autotune_cache.txt and open a PR.\n\n";
  }

  // Collect every (variant_idx, blocksz) timing across every target so we
  // can score variants holistically at the end. Indexed by [target_idx]
  // [(variant_idx, blocksz)].
  constexpr int kSweepBlocksizes[] = {64, 128, 256, 512};
  constexpr int kNumSweepBlocksizes =
      sizeof(kSweepBlocksizes) / sizeof(kSweepBlocksizes[0]);
  struct TrialKey {
    std::size_t variant_idx;
    int blocksz;
  };
  std::vector<std::vector<double>> timings(
      kAutotuneTargets.size(),
      std::vector<double>(kNumKernelVariants * kNumSweepBlocksizes,
                          std::numeric_limits<double>::infinity()));
  auto trial_slot = [](std::size_t v, int bsz_idx) {
    return v * kNumSweepBlocksizes + bsz_idx;
  };

  // Count supported (variant, blocksz) trials per target so we can show
  // accurate progress. IsSupportedKernelConfig accepts the same set for any
  // enabled variant today, but counting per target keeps the progress
  // figure honest if that gets tightened in the future.
  int total_trials = 0;
  for (const auto &t : kAutotuneTargets) {
    for (std::size_t i = 0; i < kNumKernelVariants; ++i) {
      for (int bsz : kSweepBlocksizes) {
        KernelConfig probe = GetKernelConfigForVariant(i, t.precision);
        probe.blocksz = bsz;
        if (IsSupportedKernelConfig(probe, t.precision)) {
          ++total_trials;
        }
      }
    }
  }
  int trials_done = 0;
  auto start_time = std::chrono::steady_clock::now();

  KernelConfig last_chosen{};
  for (std::size_t tidx = 0; tidx < kAutotuneTargets.size(); ++tidx) {
    const auto &t = kAutotuneTargets[tidx];
    if (verbose) {
      std::cout << "  [target " << (tidx + 1) << "/"
                << kAutotuneTargets.size() << "] "
                << ProfileTypeName(t.profile) << " "
                << PrecisionTypeName(t.precision) << ":\n";
    }
    double best_seconds = std::numeric_limits<double>::infinity();
    KernelConfig best_cfg = GetDefaultKernelConfig(t.precision);
    // Sweep (variant geometry, blocksz) jointly. Both LaunchDoTile and
    // LaunchDoTileShfl dispatch by runtime blocksz, and
    // IsSupportedKernelConfig admits 64/128/256/512 for any enabled
    // variant -- the autotuner used to pick a single precision-tied
    // blocksz (256 DP / 512 SP) per variant, hiding configs that win at
    // 128 (e.g. shfl is occupancy-bound and tighter blocks let more
    // run concurrently).
    for (std::size_t i = 0; i < kNumKernelVariants; ++i) {
      for (int bsz_idx = 0; bsz_idx < kNumSweepBlocksizes; ++bsz_idx) {
        int bsz = kSweepBlocksizes[bsz_idx];
        KernelConfig cfg = GetKernelConfigForVariant(i, t.precision);
        cfg.blocksz = bsz;
        if (!IsSupportedKernelConfig(cfg, t.precision)) {
          continue;
        }
        double seconds = std::numeric_limits<double>::infinity();
        try {
          seconds = bench(device_id, t.profile, t.precision, cfg);
        } catch (const std::exception &e) {
          if (verbose) {
            std::cout << "    v" << i << " blocksz=" << bsz
                      << " threw: " << e.what() << " (skipped)\n";
          }
          continue;
        }
        timings[tidx][trial_slot(i, bsz_idx)] = seconds;
        ++trials_done;
        if (verbose) {
          // Progress prefix: [done/total pct%] -- written first so even a
          // long-running sweep streamed to a log file is easy to monitor.
          // ETA is a coarse linear extrapolation from elapsed wall time.
          auto now = std::chrono::steady_clock::now();
          double elapsed_s =
              std::chrono::duration<double>(now - start_time).count();
          double pct = 100.0 * trials_done / std::max(total_trials, 1);
          double eta_s = elapsed_s * (total_trials - trials_done) /
                         std::max(trials_done, 1);
          std::cout << "    [" << trials_done << "/" << total_trials << " "
                    << std::fixed << std::setprecision(1) << pct
                    << "% eta=" << static_cast<int>(eta_s) << "s] "
                    << std::defaultfloat
                    << "v" << i << " blocksz=" << cfg.blocksz
                    << ": bps=" << cfg.blocks_per_sm
                    << " dpt=" << cfg.diags_per_thread
                    << " ur=" << cfg.unrolled_rows
                    << " our=" << cfg.outer_unrolled_rows
                    << " kti=" << cfg.kernel_tile_iters
                    << " (tile_height=" << cfg.tile_height() << ") -> "
                    << seconds << " s\n";
          // Force-flush so log tail -f / pipe consumers see lines live
          // rather than waiting for the per-page stdio buffer to flush.
          std::cout.flush();
        }
        if (seconds < best_seconds) {
          best_seconds = seconds;
          best_cfg = cfg;
        }
      }
    }
    disk_cache.Store(device_key, t.profile, t.precision, best_cfg);
    last_chosen = best_cfg;
    if (verbose) {
      // Identify which kVariants[] entry the winner came from so we can
      // print it by name (e.g. "v6") instead of forcing the reader to
      // match the tuple manually.
      std::size_t winner_idx = kNumKernelVariants;
      for (std::size_t i = 0; i < kNumKernelVariants; ++i) {
        const auto &v = GetKernelVariantGeometry(i);
        if (v.blocks_per_sm == best_cfg.blocks_per_sm &&
            v.diags_per_thread == best_cfg.diags_per_thread &&
            v.unrolled_rows == best_cfg.unrolled_rows &&
            v.outer_unrolled_rows == best_cfg.outer_unrolled_rows &&
            v.kernel_tile_iters == best_cfg.kernel_tile_iters) {
          winner_idx = i;
          break;
        }
      }
      std::cout << "    WINNER: ";
      if (winner_idx < kNumKernelVariants) {
        std::cout << "v" << winner_idx << " ";
      }
      std::cout << "blocksz=" << best_cfg.blocksz
                << " (bps=" << best_cfg.blocks_per_sm
                << " dpt=" << best_cfg.diags_per_thread
                << " ur=" << best_cfg.unrolled_rows
                << " our=" << best_cfg.outer_unrolled_rows
                << " kti=" << best_cfg.kernel_tile_iters << ") -> "
                << best_seconds << " s\n\n";
    }
  }

  disk_cache.Save();

  // Invalidate the process-wide cached copy of the user override so subsequent
  // kernel launches in the same process pick up the fresh entries.
  {
    std::lock_guard<std::mutex> lock(SharedCacheMutex());
    SharedCacheState().user_cache.reset();
    SharedCacheState().loaded_user_path.clear();
  }

  if (verbose) {
    // Cross-target variant score: for each (variant, blocksz) trial, compute
    // its time RELATIVE TO the best trial in the same (profile, precision)
    // target -- so the winner of that target gets 1.0, others get >1.0 in
    // proportion to how far behind they are. The trial's overall score is
    // the geometric mean of those relative times across all targets where it
    // ran successfully; geomean is the right aggregate because it penalizes
    // a single really-bad target much more than a single really-good one
    // (the user explicitly asked for this: "wins really well once but
    // scores really poorly in other cases isn't a good default").
    //
    // Lower score = more robust default. A trial with score 1.05 is on
    // average 5% slower than the per-target winner.
    struct TrialScore {
      std::size_t variant_idx;
      int blocksz;
      double geomean_ratio;
      int num_targets;
      int num_wins;
      double worst_ratio;
    };
    std::vector<TrialScore> trial_scores;
    trial_scores.reserve(kNumKernelVariants * kNumSweepBlocksizes);

    // Per-target best, used as denominator.
    std::vector<double> per_target_best(kAutotuneTargets.size(),
                                        std::numeric_limits<double>::infinity());
    for (std::size_t tidx = 0; tidx < kAutotuneTargets.size(); ++tidx) {
      for (double t_s : timings[tidx]) {
        if (t_s < per_target_best[tidx]) per_target_best[tidx] = t_s;
      }
    }
    for (std::size_t i = 0; i < kNumKernelVariants; ++i) {
      for (int bsz_idx = 0; bsz_idx < kNumSweepBlocksizes; ++bsz_idx) {
        double log_sum = 0.0;
        int n = 0;
        int wins = 0;
        double worst_ratio = 1.0;
        for (std::size_t tidx = 0; tidx < kAutotuneTargets.size(); ++tidx) {
          double t_s = timings[tidx][trial_slot(i, bsz_idx)];
          if (!std::isfinite(t_s)) continue;
          double best_s = per_target_best[tidx];
          if (!std::isfinite(best_s) || best_s <= 0.0) continue;
          double ratio = t_s / best_s;
          log_sum += std::log(ratio);
          ++n;
          if (ratio == 1.0) ++wins;
          if (ratio > worst_ratio) worst_ratio = ratio;
        }
        if (n == 0) continue;
        double geomean = std::exp(log_sum / static_cast<double>(n));
        trial_scores.push_back({i, kSweepBlocksizes[bsz_idx], geomean, n, wins,
                                worst_ratio});
      }
    }
    std::sort(trial_scores.begin(), trial_scores.end(),
              [](const TrialScore &a, const TrialScore &b) {
                return a.geomean_ratio < b.geomean_ratio;
              });

    std::cout << "  Cross-target variant score (lower = better default;\n"
              << "  1.00 = always tied with per-target winner, 2.00 = on\n"
              << "  geomean average 2x slower than per-target winner):\n";
    std::cout << "    rank  trial             geomean  worst  wins/targets\n";
    std::cout << "    ----  ----------------  -------  -----  ------------\n";
    int rank = 1;
    for (const auto &s : trial_scores) {
      std::cout << "    " << std::setw(4) << rank << "  "
                << "v" << s.variant_idx << " bsz=" << std::setw(3)
                << s.blocksz << "        "
                << std::fixed << std::setprecision(3) << s.geomean_ratio
                << "    " << std::setprecision(2) << s.worst_ratio
                << "    " << s.num_wins << "/" << s.num_targets << "\n";
      ++rank;
    }
    std::cout << std::defaultfloat;
    if (!trial_scores.empty()) {
      const auto &top = trial_scores.front();
      std::cout << "\n  RECOMMENDED DEFAULT: v" << top.variant_idx
                << " blocksz=" << top.blocksz << " (geomean " << std::fixed
                << std::setprecision(3) << top.geomean_ratio
                << "x of per-target best, worst-case " << std::setprecision(2)
                << top.worst_ratio << "x)\n" << std::defaultfloat;
    }
    std::cout << "\n";
    std::cout << "  wrote " << kAutotuneTargets.size() << " entries to "
              << resolved << std::endl;
  }
  return AutotuneResult{device_key, resolved, last_chosen, true};
}

KernelConfig GetKernelConfigForDevice(int device_id,
                                      SCAMPProfileType profile_type,
                                      SCAMPPrecisionType precision,
                                      const std::string &cache_path) {
  // 0) Per-thread override. Set by the autotune benchmark loop to force a
  //    specific variant on a per-timed-run basis; bypasses the cache.
  if (auto override = GetKernelConfigOverride(); override.has_value()) {
    return *override;
  }
  KernelConfig fallback = GetDefaultKernelConfig(precision);
  try {
    GpuDeviceProps props = QueryDeviceProps(device_id);
    std::string key = props.CacheKey();

    std::lock_guard<std::mutex> lock(SharedCacheMutex());

    // 1) User override (env var or ~/.cache/scamp/autotune.txt).
    const AutotuneCache *user = GetOrLoadUserCache(cache_path);
    auto user_hit = user->Lookup(key, profile_type, precision);
    if (user_hit.has_value() && IsSupportedKernelConfig(*user_hit, precision)) {
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
