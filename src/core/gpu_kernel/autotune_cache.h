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
//   <device_key>|<profile_type>|<precision>|<blocksz>|<tile_height>|<blocks_per_sm>
//   ...
//
// Comment lines starting with '#' and blank lines are ignored.
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
