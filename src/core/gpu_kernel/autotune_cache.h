#pragma once
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>
#include "common/common.h"
#include "kernel_config.h"

namespace SCAMP {

// AutotuneCache is the on-disk record of which KernelConfig the autotuner
// has picked for each (device, profile_type, precision) tuple.
//
// File format is a small plain-text format (one record per line, '|'
// separated) so that we don't need a JSON dependency and so that a user can
// hand-inspect or hand-edit the cache when debugging. The header line is a
// version marker; future format changes bump it and ignore older files.
//
//   SCAMP_AUTOTUNE_V1
//   <device_key>|<profile_type>|<precision>|<blocksz>|<blocks_per_sm>
//     |<diags_per_thread>|<unrolled_rows>|<outer_unrolled_rows>
//     |<kernel_tile_iters>
//   ...
//
// Comment lines starting with '#' and blank lines are ignored.
//
// ============================================================
// Upgrade compatibility (when SCAMP_VARIANT_TUPLES changes, or the
// kernel implementation changes such that an old tuned config is no
// longer best):
// ============================================================
//
// The intent is "users keep their existing tuned cache by default, and
// only re-tune when they explicitly want to or when a release forces
// it." Concretely:
//
//   - Add / remove / reorder a variant in SCAMP_VARIANT_TUPLES:
//     no version bump needed. The runtime lookup matches by
//     (bps,dpt,ur,our,kti) tuple and falls through to the next cache
//     source on an unsupported tuple (see
//     LookupKernelConfigForDeviceKey + IsSupportedKernelConfig).
//     Per-(device, profile, precision) entries are independent --
//     editing the variant table only invalidates the cache entries
//     that named a now-removed tuple; everything else still hits.
//
//   - Change the on-disk record schema (e.g. add a column):
//     bump kHeader from SCAMP_AUTOTUNE_V<N> to V<N+1>. ParseStream
//     silently treats a non-matching header as an empty cache, so an
//     end-user upgrading their pyscamp pip wheel won't see a
//     SCAMPException -- they just fall through to the new release's
//     built-in cache (or the cache-miss warning) and can re-tune at
//     their leisure.
//
//   - Change kernel semantics with the SAME tuple still valid (e.g.
//     refactor do_iteration_fast so a "DPT=4 OUR=16" kernel emits
//     materially different instructions): the existing cache will
//     still match by tuple, but the recorded winner may no longer be
//     the actual fastest cfg. If the regression is meaningful, bump
//     kHeader to force everyone to re-tune; if it isn't, leave the
//     header alone and let users opt-in to re-tuning.
//
// Each of the cases above is covered by a unit test in
// test/cpp/test_autotune_cache.cpp (Test_UnsupportedVariantInCache...,
// Test_FutureVersionHeaderSilentlyEmpties,
// Test_PartiallyStaleCachePreservesGoodEntries,
// Test_StaleUserCacheFallsThroughToBuiltin). Adding a new failure
// mode to the cache should land with a matching test there.
//
// Cache file location (in priority order):
//   1. Path passed to the constructor explicitly
//   2. $SCAMP_AUTOTUNE_CACHE if set
//   3. $XDG_CACHE_HOME/scamp/autotune.txt
//   4. Platform-specific user dir:
//        Linux/macOS: $HOME/.cache/scamp/autotune.txt
//        Windows:     %LOCALAPPDATA%\scamp\autotune.txt
//                     (falls back to %USERPROFILE%\.cache\scamp\autotune.txt
//                      when LOCALAPPDATA is unset, for layout symmetry with
//                      POSIX hosts).
class AutotuneCache {
 public:
  // Resolve the default cache path according to the rules above.
  static std::string DefaultPath();

  // Construct an empty in-memory cache. Use Load()/Save() to read or write.
  AutotuneCache();
  explicit AutotuneCache(std::string path);

  // Returns true if a cache file exists on disk at this path.
  bool FileExists() const;

  // Read entries from disk. If the file does not exist or is empty this is a
  // no-op (an empty cache is a valid state). Throws on malformed input.
  void Load();

  // Parse cache entries from an in-memory string instead of disk. Used to
  // load the binary's built-in autotune cache (see builtin_autotune_cache.h).
  // Does not touch path(); a subsequent Save() will still write to path().
  void LoadFromString(const std::string &contents);

  // Atomically write the current in-memory entries to disk, creating parent
  // directories as needed. Throws on I/O failure.
  void Save() const;

  // Returns the cached KernelConfig for the given tuple, or std::nullopt if
  // none is recorded. The caller is responsible for falling back to
  // GetDefaultKernelConfig() in that case.
  std::optional<KernelConfig> Lookup(const std::string &device_key,
                                     SCAMPProfileType profile_type,
                                     SCAMPPrecisionType precision) const;

  // Store (or replace) a config in the in-memory cache. Save() must be called
  // separately to persist to disk.
  void Store(const std::string &device_key, SCAMPProfileType profile_type,
             SCAMPPrecisionType precision, const KernelConfig &cfg);

  // Path on disk this cache reads from / writes to.
  const std::string &path() const { return path_; }

  // Number of entries currently in the in-memory cache.
  size_t size() const { return entries_.size(); }

 private:
  struct Key {
    std::string device_key;
    SCAMPProfileType profile_type;
    SCAMPPrecisionType precision;
    bool operator==(const Key &o) const {
      return device_key == o.device_key && profile_type == o.profile_type &&
             precision == o.precision;
    }
  };
  struct KeyHash {
    size_t operator()(const Key &k) const noexcept {
      size_t h = std::hash<std::string>{}(k.device_key);
      h = h * 31 + std::hash<int>{}(static_cast<int>(k.profile_type));
      h = h * 31 + std::hash<int>{}(static_cast<int>(k.precision));
      return h;
    }
  };
  // Shared parser used by Load() (file) and LoadFromString() (memory).
  // source_label is used in error messages.
  void ParseStream(std::istream &in, const std::string &source_label);

  std::string path_;
  std::unordered_map<Key, KernelConfig, KeyHash> entries_;
};

// String <-> enum helpers used by both the cache serializer and by user-
// facing autotune output. ProfileTypeName returns a stable identifier
// suitable for the cache; ParseProfileTypeName is its inverse.
const char *ProfileTypeName(SCAMPProfileType t);
const char *PrecisionTypeName(SCAMPPrecisionType t);
bool ParseProfileTypeName(const std::string &s, SCAMPProfileType *out);
bool ParsePrecisionTypeName(const std::string &s, SCAMPPrecisionType *out);

}  // namespace SCAMP
