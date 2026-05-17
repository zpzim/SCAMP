#include "autotune_cache.h"

#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/scamp_exception.h"

namespace SCAMP {

namespace {

constexpr const char *kHeader = "SCAMP_AUTOTUNE_V1";
constexpr char kFieldSep = '|';

// Fields per record line:
//   device_key | profile_type | precision | blocksz | blocks_per_sm |
//   diags_per_thread | unrolled_rows | outer_unrolled_rows |
//   kernel_tile_iters
// (tile_height is derived as kernel_tile_iters * outer_unrolled_rows and
//  not stored separately.)
//
// No format version bump despite the field-count change: no shipped
// production cache had populated entries to migrate, so any local V1
// 6-field cache will trip SplitN and throw; the autotune lookup catches
// it and falls back to the default. Re-running RunAutotune rewrites the
// cache in the current format.
constexpr size_t kNumRecordFields = 9;

// Recursively mkdir -p. Returns 0 on success, errno on failure.
int MkdirP(const std::string &path) {
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    cur.push_back(path[i]);
    if (path[i] == '/' && !cur.empty() && cur != "/") {
      if (::mkdir(cur.c_str(), 0700) != 0 && errno != EEXIST) {
        return errno;
      }
    }
  }
  if (!path.empty() && path.back() != '/') {
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
      return errno;
    }
  }
  return 0;
}

std::string ParentDir(const std::string &path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

// Split a string on a delimiter into exactly n parts. Returns false if the
// count does not match.
bool SplitN(const std::string &s, char delim, size_t n,
            std::vector<std::string> *out) {
  out->clear();
  out->reserve(n);
  std::stringstream ss(s);
  std::string field;
  while (std::getline(ss, field, delim)) {
    out->push_back(std::move(field));
  }
  return out->size() == n;
}

}  // namespace

const char *ProfileTypeName(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_1NN:
      return "1NN";
    case PROFILE_TYPE_1NN_INDEX:
      return "1NN_INDEX";
    case PROFILE_TYPE_SUM_THRESH:
      return "SUM_THRESH";
    case PROFILE_TYPE_FREQUENCY_THRESH:
      return "FREQUENCY_THRESH";
    case PROFILE_TYPE_KNN:
      return "KNN";
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return "APPROX_ALL_NEIGHBORS";
    case PROFILE_TYPE_MATRIX_SUMMARY:
      return "MATRIX_SUMMARY";
    case PROFILE_TYPE_1NN_MULTIDIM:
      return "1NN_MULTIDIM";
    case PROFILE_TYPE_INVALID:
    default:
      return "INVALID";
  }
}

const char *PrecisionTypeName(SCAMPPrecisionType t) {
  switch (t) {
    case PRECISION_SINGLE:
      return "SINGLE";
    case PRECISION_MIXED:
      return "MIXED";
    case PRECISION_DOUBLE:
      return "DOUBLE";
    case PRECISION_ULTRA:
      return "ULTRA";
    case PRECISION_INVALID:
    default:
      return "INVALID";
  }
}

bool ParseProfileTypeName(const std::string &s, SCAMPProfileType *out) {
  // The enum is not contiguous and 1NN/MATRIX_SUMMARY/APPROX_ALL_NEIGHBORS sit
  // above the prior PROFILE_TYPE_1NN_MULTIDIM upper bound; iterate the full
  // range to cover them. (Out-of-range integer values fall through to
  // ProfileTypeName's default "INVALID" branch and never match an input
  // string other than literally "INVALID".)
  for (int i = static_cast<int>(PROFILE_TYPE_INVALID);
       i <= static_cast<int>(PROFILE_TYPE_MATRIX_SUMMARY); ++i) {
    auto candidate = static_cast<SCAMPProfileType>(i);
    if (s == ProfileTypeName(candidate)) {
      *out = candidate;
      return true;
    }
  }
  return false;
}

bool ParsePrecisionTypeName(const std::string &s, SCAMPPrecisionType *out) {
  for (int i = static_cast<int>(PRECISION_INVALID);
       i <= static_cast<int>(PRECISION_ULTRA); ++i) {
    auto candidate = static_cast<SCAMPPrecisionType>(i);
    if (s == PrecisionTypeName(candidate)) {
      *out = candidate;
      return true;
    }
  }
  return false;
}

std::string AutotuneCache::DefaultPath() {
  if (const char *override = std::getenv("SCAMP_AUTOTUNE_CACHE")) {
    return override;
  }
  if (const char *xdg = std::getenv("XDG_CACHE_HOME")) {
    return std::string(xdg) + "/scamp/autotune.txt";
  }
  if (const char *home = std::getenv("HOME")) {
    return std::string(home) + "/.cache/scamp/autotune.txt";
  }
  // Last resort: write to current directory.
  return "./scamp_autotune.txt";
}

AutotuneCache::AutotuneCache() : path_(DefaultPath()) {}

AutotuneCache::AutotuneCache(std::string path) : path_(std::move(path)) {}

bool AutotuneCache::FileExists() const {
  struct stat st{};
  return ::stat(path_.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void AutotuneCache::LoadFromString(const std::string &contents) {
  entries_.clear();
  std::stringstream in(contents);
  ParseStream(in, "<builtin>");
}

void AutotuneCache::Load() {
  entries_.clear();
  std::ifstream in(path_);
  if (!in.is_open()) {
    // Missing file is a valid "empty cache" state.
    return;
  }
  ParseStream(in, path_);
}

void AutotuneCache::ParseStream(std::istream &in,
                                const std::string &source_label) {
  std::string line;
  bool header_seen = false;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty()) continue;
    if (line[0] == '#') continue;
    if (!header_seen) {
      if (line != kHeader) {
        // Unknown format -- treat as empty rather than corrupt the user's
        // workflow. Future format bumps will land here.
        entries_.clear();
        return;
      }
      header_seen = true;
      continue;
    }

    std::vector<std::string> fields;
    if (!SplitN(line, kFieldSep, kNumRecordFields, &fields)) {
      throw SCAMPException("Malformed autotune cache at " + source_label + ":" +
                           std::to_string(lineno));
    }

    Key key;
    key.device_key = fields[0];
    if (!ParseProfileTypeName(fields[1], &key.profile_type)) {
      throw SCAMPException("Unknown profile type '" + fields[1] +
                           "' in autotune cache at " + source_label + ":" +
                           std::to_string(lineno));
    }
    if (!ParsePrecisionTypeName(fields[2], &key.precision)) {
      throw SCAMPException("Unknown precision '" + fields[2] +
                           "' in autotune cache at " + source_label + ":" +
                           std::to_string(lineno));
    }

    KernelConfig cfg{};
    try {
      cfg.blocksz = std::stoi(fields[3]);
      cfg.blocks_per_sm = std::stoi(fields[4]);
      cfg.diags_per_thread = std::stoi(fields[5]);
      cfg.unrolled_rows = std::stoi(fields[6]);
      cfg.outer_unrolled_rows = std::stoi(fields[7]);
      cfg.kernel_tile_iters = std::stoi(fields[8]);
    } catch (const std::exception &e) {
      throw SCAMPException("Malformed integer in autotune cache at " +
                           source_label + ":" + std::to_string(lineno) +
                           " -- " + e.what());
    }

    entries_[key] = cfg;
  }
}

void AutotuneCache::Save() const {
  std::string parent = ParentDir(path_);
  if (parent != "." && parent != "/") {
    int err = MkdirP(parent);
    if (err != 0) {
      throw SCAMPException("Cannot create autotune cache directory '" + parent +
                           "': " + std::strerror(err));
    }
  }

  // Atomic write: write to a temp file in the same directory and rename.
  std::string tmp = path_ + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out.is_open()) {
      throw SCAMPException("Cannot open autotune cache for writing: " + tmp);
    }
    out << kHeader << "\n";
    out << "# device_key|profile_type|precision|blocksz|blocks_per_sm|"
           "diags_per_thread|unrolled_rows|outer_unrolled_rows|"
           "kernel_tile_iters\n";
    out << "# (tile_height = kernel_tile_iters * outer_unrolled_rows)\n";
    for (const auto &kv : entries_) {
      const Key &k = kv.first;
      const KernelConfig &c = kv.second;
      out << k.device_key << kFieldSep << ProfileTypeName(k.profile_type)
          << kFieldSep << PrecisionTypeName(k.precision) << kFieldSep
          << c.blocksz << kFieldSep << c.blocks_per_sm << kFieldSep
          << c.diags_per_thread << kFieldSep << c.unrolled_rows << kFieldSep
          << c.outer_unrolled_rows << kFieldSep << c.kernel_tile_iters << "\n";
    }
    out.flush();
    if (out.fail()) {
      throw SCAMPException("Write to autotune cache failed: " + tmp);
    }
  }
  if (std::rename(tmp.c_str(), path_.c_str()) != 0) {
    throw SCAMPException("Cannot rename " + tmp + " to " + path_ + ": " +
                         std::strerror(errno));
  }
}

std::optional<KernelConfig> AutotuneCache::Lookup(
    const std::string &device_key, SCAMPProfileType profile_type,
    SCAMPPrecisionType precision) const {
  Key k{device_key, profile_type, precision};
  auto it = entries_.find(k);
  if (it == entries_.end()) return std::nullopt;
  return it->second;
}

void AutotuneCache::Store(const std::string &device_key,
                          SCAMPProfileType profile_type,
                          SCAMPPrecisionType precision,
                          const KernelConfig &cfg) {
  Key k{device_key, profile_type, precision};
  entries_[k] = cfg;
}

}  // namespace SCAMP
