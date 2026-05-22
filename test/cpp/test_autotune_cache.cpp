// Unit tests for SCAMP's autotune cache lookup chain.
//
// These tests run WITHOUT a real CUDA device or runtime -- they drive
// `LookupKernelConfigForDeviceKey` directly with synthetic device keys and
// in-memory AutotuneCache instances. The on-disk path resolution is
// exercised separately by reading the cache file the test creates in a
// tmp dir.
//
// Tiny in-file harness (no gtest dep). Failures bump a counter and the
// process exits non-zero so ctest reports the failure. Each test is a
// function called from main(); add new ones the same way.

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/gpu_kernel/autotune.h"
#include "core/gpu_kernel/autotune_cache.h"
#include "core/gpu_kernel/kernel_config.h"

namespace {

// Portable setenv/unsetenv: MSVC's CRT has no setenv()/unsetenv() but
// provides _putenv_s(). Use those under _WIN32; everything else gets the
// POSIX implementations.
int PortableSetenv(const char *name, const char *value) {
#ifdef _WIN32
  return _putenv_s(name, value);
#else
  return setenv(name, value, /*overwrite=*/1);
#endif
}

int PortableUnsetenv(const char *name) {
#ifdef _WIN32
  // _putenv_s with an empty string removes the variable.
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

// Returns a tmp file path unique to this process (counter increments on
// each call). Used so concurrent test runs don't collide. We avoid
// getpid() because MSVC's CRT spells it _getpid(); std::filesystem and
// a static counter give the same uniqueness without the ifdef churn.
std::string MakeTempPath(const char *stem) {
  static int counter = 0;
  ++counter;
  std::filesystem::path p = std::filesystem::temp_directory_path();
  p /= std::string("scamp_test_") + stem + "_" +
       std::to_string(counter) + ".txt";
  return p.string();
}

int g_failures = 0;
const char *g_current_test = nullptr;

#define EXPECT_TRUE(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::cerr << "[FAIL] " << g_current_test << " @ " << __FILE__ << ":" \
                << __LINE__ << ": EXPECT_TRUE(" #cond ")\n";               \
    }                                                                      \
  } while (0)

#define EXPECT_EQ(a, b)                                                    \
  do {                                                                     \
    auto _va = (a);                                                        \
    auto _vb = (b);                                                        \
    if (!(_va == _vb)) {                                                   \
      ++g_failures;                                                        \
      std::cerr << "[FAIL] " << g_current_test << " @ " << __FILE__ << ":" \
                << __LINE__ << ": EXPECT_EQ(" #a ", " #b "): got " << _va  \
                << " vs " << _vb << "\n";                                  \
    }                                                                      \
  } while (0)

// Capture stderr from a callable. Tests use this to inspect cache-miss
// warning output without polluting the test runner's stderr.
template <typename F>
std::string CaptureStderr(F &&f) {
  std::stringstream buf;
  std::streambuf *old = std::cerr.rdbuf(buf.rdbuf());
  try {
    f();
  } catch (...) {
    std::cerr.rdbuf(old);
    throw;
  }
  std::cerr.rdbuf(old);
  return buf.str();
}

// Build an in-memory AutotuneCache from a string in the on-disk format.
// Returns a heap-allocated cache; caller owns. Wraps LoadFromString to
// keep test bodies short.
std::unique_ptr<SCAMP::AutotuneCache> MakeCacheFromString(
    const std::string &contents) {
  auto c = std::make_unique<SCAMP::AutotuneCache>();
  c->LoadFromString(contents);
  return c;
}

// A KernelConfig that DOES match a registered kVariants entry (v6) so it
// passes IsSupportedKernelConfig. We use this as the "hit" config in
// positive tests.
SCAMP::KernelConfig SupportedConfig() {
  SCAMP::KernelConfig c{};
  c.blocksz = 128;
  c.blocks_per_sm = 8;
  c.diags_per_thread = 4;
  c.unrolled_rows = 0;
  c.outer_unrolled_rows = 8;
  c.kernel_tile_iters = 8;
  return c;
}

SCAMP::KernelConfig FallbackConfig() {
  // Anything goes -- the lookup chain just returns it on miss without
  // checking IsSupportedKernelConfig.
  SCAMP::KernelConfig c{};
  c.blocksz = 256;
  c.blocks_per_sm = 2;
  c.diags_per_thread = 2;
  c.unrolled_rows = 2;
  c.outer_unrolled_rows = 16;
  c.kernel_tile_iters = 16;
  return c;
}

bool ConfigsEqual(const SCAMP::KernelConfig &a, const SCAMP::KernelConfig &b) {
  return a.blocksz == b.blocksz && a.blocks_per_sm == b.blocks_per_sm &&
         a.diags_per_thread == b.diags_per_thread &&
         a.unrolled_rows == b.unrolled_rows &&
         a.outer_unrolled_rows == b.outer_unrolled_rows &&
         a.kernel_tile_iters == b.kernel_tile_iters;
}

// ===========================================================================
// Tests
// ===========================================================================

// (Contract #1) Cache hit on a user override returns the cached config.
void Test_UserCacheHit() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "FakeDeviceA|1NN_INDEX|SINGLE|128|8|4|0|8|8\n");
  std::string out = CaptureStderr([&]() {
    auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
        "FakeDeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        user.get(), nullptr, FallbackConfig());
    EXPECT_TRUE(ConfigsEqual(cfg, SupportedConfig()));
  });
  EXPECT_TRUE(out.empty());  // no warning expected on hit
}

// (Contract #1) Cache hit on the built-in cache returns the cached config.
void Test_BuiltinCacheHit() {
  SCAMP::ResetAutotuneWarnings();
  auto builtin = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "FakeDeviceA|1NN_INDEX|SINGLE|128|8|4|0|8|8\n");
  std::string out = CaptureStderr([&]() {
    auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
        "FakeDeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        nullptr, builtin.get(), FallbackConfig());
    EXPECT_TRUE(ConfigsEqual(cfg, SupportedConfig()));
  });
  EXPECT_TRUE(out.empty());
}

// (Contract #1) User override wins over built-in.
void Test_UserCacheBeatsBuiltin() {
  SCAMP::ResetAutotuneWarnings();
  // User cache picks variant 6 (matches SupportedConfig); built-in
  // picks variant 0 (compile-time default in our fixture). The user
  // entry should win.
  auto user = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "FakeDeviceA|1NN_INDEX|SINGLE|128|8|4|0|8|8\n");
  auto builtin = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "FakeDeviceA|1NN_INDEX|SINGLE|256|2|2|2|16|16\n");
  auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
      "FakeDeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
      user.get(), builtin.get(), FallbackConfig());
  EXPECT_TRUE(ConfigsEqual(cfg, SupportedConfig()));  // = the user entry
}

// (Contract #2) Cache miss returns fallback and emits a one-shot warning.
void Test_CacheMissEmitsWarningAndReturnsFallback() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString("SCAMP_AUTOTUNE_V1\n");  // empty cache
  std::string out = CaptureStderr([&]() {
    auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDevice", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        user.get(), nullptr, FallbackConfig());
    EXPECT_TRUE(ConfigsEqual(cfg, FallbackConfig()));
  });
  EXPECT_TRUE(out.find("UnknownDevice") != std::string::npos);
  EXPECT_TRUE(out.find("1NN_INDEX") != std::string::npos);
  EXPECT_TRUE(out.find("SINGLE") != std::string::npos);
  EXPECT_TRUE(out.find("--autotune") != std::string::npos);
}

// (Contract #2) Repeated miss for the same tuple is silent the second time.
void Test_CacheMissWarningIsOneShot() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString("SCAMP_AUTOTUNE_V1\n");
  std::string first = CaptureStderr([&]() {
    SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDevice", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        user.get(), nullptr, FallbackConfig());
  });
  std::string second = CaptureStderr([&]() {
    SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDevice", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        user.get(), nullptr, FallbackConfig());
  });
  EXPECT_TRUE(!first.empty());
  EXPECT_TRUE(second.empty());  // warning should be deduped
}

// (Contract #2) SCAMP_AUTOTUNE_QUIET=1 suppresses the cache-miss warning.
// pyscamp sets this at module init time so notebook users don't get
// uninvited stderr output.
void Test_QuietEnvSuppressesWarning() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString("SCAMP_AUTOTUNE_V1\n");
  PortableSetenv("SCAMP_AUTOTUNE_QUIET", "1");
  std::string out = CaptureStderr([&]() {
    auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDeviceQuiet", SCAMP::PROFILE_TYPE_1NN_INDEX,
        SCAMP::PRECISION_SINGLE, user.get(), nullptr, FallbackConfig());
    EXPECT_TRUE(ConfigsEqual(cfg, FallbackConfig()));
  });
  EXPECT_TRUE(out.empty());
  PortableUnsetenv("SCAMP_AUTOTUNE_QUIET");
}

// (Contract #2) Edge-case env values: "0" and "false" mean DO emit the
// warning; only truthy non-empty values silence it.
void Test_QuietEnv_FalseyValuesStillWarn() {
  for (const char *val : {"0", "false", ""}) {
    SCAMP::ResetAutotuneWarnings();
    auto user = MakeCacheFromString("SCAMP_AUTOTUNE_V1\n");
    PortableSetenv("SCAMP_AUTOTUNE_QUIET", val);
    std::string out = CaptureStderr([&]() {
      SCAMP::LookupKernelConfigForDeviceKey(
          "FalseyEnvCheck", SCAMP::PROFILE_TYPE_1NN_INDEX,
          SCAMP::PRECISION_SINGLE, user.get(), nullptr, FallbackConfig());
    });
    EXPECT_TRUE(!out.empty());
    PortableUnsetenv("SCAMP_AUTOTUNE_QUIET");
  }
}

// (Contract #2) A *different* (profile, precision) for the same device
// re-arms the warning (it's a different tuple).
void Test_CacheMissWarningIsPerTuple() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString("SCAMP_AUTOTUNE_V1\n");
  std::string a = CaptureStderr([&]() {
    SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDevice", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
        user.get(), nullptr, FallbackConfig());
  });
  std::string b = CaptureStderr([&]() {
    SCAMP::LookupKernelConfigForDeviceKey(
        "UnknownDevice", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_DOUBLE,
        user.get(), nullptr, FallbackConfig());
  });
  EXPECT_TRUE(!a.empty());
  EXPECT_TRUE(!b.empty());  // different precision should emit a new warning
}

// (Contract #1) An unsupported variant tuple in the cache (i.e. the
// (bps,dpt,ur,our,kti) tuple doesn't match any registered kVariants
// entry) is rejected by IsSupportedKernelConfig and falls through to
// the next source. We use a clearly-bogus tuple so this stays robust if
// kVariants changes later.
void Test_UnsupportedVariantInCacheFallsThrough() {
  SCAMP::ResetAutotuneWarnings();
  auto user = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "FakeDeviceA|1NN_INDEX|SINGLE|128|99|99|99|99|99\n");
  auto cfg = SCAMP::LookupKernelConfigForDeviceKey(
      "FakeDeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
      user.get(), nullptr, FallbackConfig());
  EXPECT_TRUE(ConfigsEqual(cfg, FallbackConfig()));
}

// Multi-device cache: looking up device A returns A's entry, device B
// returns B's entry. Regression for the "P100 entries clobbered 3080
// entries" bug-class.
void Test_MultiDeviceCacheLookup() {
  SCAMP::ResetAutotuneWarnings();
  auto cache = MakeCacheFromString(
      "SCAMP_AUTOTUNE_V1\n"
      "DeviceA|1NN_INDEX|SINGLE|128|8|4|0|8|8\n"
      "DeviceB|1NN_INDEX|SINGLE|256|2|2|2|16|16\n");
  auto a = SCAMP::LookupKernelConfigForDeviceKey(
      "DeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
      cache.get(), nullptr, FallbackConfig());
  auto b = SCAMP::LookupKernelConfigForDeviceKey(
      "DeviceB", SCAMP::PROFILE_TYPE_1NN_INDEX, SCAMP::PRECISION_SINGLE,
      cache.get(), nullptr, FallbackConfig());
  EXPECT_EQ(a.blocksz, 128);
  EXPECT_EQ(b.blocksz, 256);
}

// (Contract #3) DefaultPath honors $SCAMP_AUTOTUNE_CACHE when set. The
// override path is opaque -- DefaultPath returns it verbatim regardless of
// platform, so we use a tmp file path that's valid on both POSIX and
// Windows.
void Test_DefaultPath_EnvVarOverride() {
  std::string override_path = MakeTempPath("override");
  PortableSetenv("SCAMP_AUTOTUNE_CACHE", override_path.c_str());
  std::string p = SCAMP::AutotuneCache::DefaultPath();
  EXPECT_EQ(p, override_path);
  PortableUnsetenv("SCAMP_AUTOTUNE_CACHE");
}

// (Contract #3) DefaultPath honors $XDG_CACHE_HOME when SCAMP_AUTOTUNE_CACHE
// is absent. XDG_CACHE_HOME is honored on all platforms (it's a SCAMP
// convention here, not a strict XDG-base-dir-spec compliance), so this
// test asserts the same layout on Windows too.
void Test_DefaultPath_XdgCacheHome() {
  PortableUnsetenv("SCAMP_AUTOTUNE_CACHE");
  std::filesystem::path xdg = std::filesystem::temp_directory_path() /
                              "scamp_test_xdg_cache";
  PortableSetenv("XDG_CACHE_HOME", xdg.string().c_str());
  std::string p = SCAMP::AutotuneCache::DefaultPath();
  std::filesystem::path expected = xdg / "scamp" / "autotune.txt";
  // DefaultPath builds the XDG path with '/' on both platforms (it's a
  // POSIX-style convention); compare via filesystem::path so '/' vs '\\'
  // normalization doesn't matter.
  EXPECT_TRUE(std::filesystem::path(p).lexically_normal() ==
              expected.lexically_normal());
  PortableUnsetenv("XDG_CACHE_HOME");
}

// (Contract #3) DefaultPath falls back to a platform-appropriate user dir
// when neither override nor XDG_CACHE_HOME is set. On POSIX this is
// $HOME/.cache/scamp/autotune.txt; on Windows it's LOCALAPPDATA/scamp/
// autotune.txt. The test sets the relevant env var to a tmp dir, runs
// DefaultPath(), and asserts the result equals the expected layout.
void Test_DefaultPath_UserDirFallback() {
  PortableUnsetenv("SCAMP_AUTOTUNE_CACHE");
  PortableUnsetenv("XDG_CACHE_HOME");
  std::filesystem::path tmp_user_dir =
      std::filesystem::temp_directory_path() / "scamp_test_user_dir";
#ifdef _WIN32
  PortableSetenv("LOCALAPPDATA", tmp_user_dir.string().c_str());
  std::filesystem::path expected =
      tmp_user_dir / "scamp" / "autotune.txt";
#else
  PortableSetenv("HOME", tmp_user_dir.string().c_str());
  std::filesystem::path expected =
      tmp_user_dir / ".cache" / "scamp" / "autotune.txt";
#endif
  std::string p = SCAMP::AutotuneCache::DefaultPath();
  EXPECT_TRUE(std::filesystem::path(p).lexically_normal() ==
              expected.lexically_normal());
#ifdef _WIN32
  PortableUnsetenv("LOCALAPPDATA");
#else
  PortableUnsetenv("HOME");
#endif
}

// Round-trip: write a cache to disk, read it back, lookup should hit.
void Test_DiskRoundTrip() {
  SCAMP::ResetAutotuneWarnings();
  std::string path = MakeTempPath("disk_roundtrip");

  SCAMP::AutotuneCache writer(path);
  SCAMP::KernelConfig cfg = SupportedConfig();
  writer.Store("DeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX,
               SCAMP::PRECISION_SINGLE, cfg);
  writer.Save();

  SCAMP::AutotuneCache reader(path);
  reader.Load();
  auto found = reader.Lookup("DeviceA", SCAMP::PROFILE_TYPE_1NN_INDEX,
                             SCAMP::PRECISION_SINGLE);
  EXPECT_TRUE(found.has_value());
  if (found.has_value()) {
    EXPECT_TRUE(ConfigsEqual(*found, cfg));
  }
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(path), ec);
}

// Save() must create intermediate directories. Exercise the
// create_directories code path by writing into a tmp subdir that doesn't
// yet exist.
void Test_SaveCreatesParentDirectories() {
  SCAMP::ResetAutotuneWarnings();
  std::filesystem::path nested =
      std::filesystem::temp_directory_path() / "scamp_test_mkdir" /
      "deep" / "nested";
  std::error_code ec;
  std::filesystem::remove_all(nested.parent_path().parent_path(), ec);
  std::filesystem::path file = nested / "autotune.txt";

  SCAMP::AutotuneCache writer(file.string());
  writer.Store("DeviceMkdir", SCAMP::PROFILE_TYPE_1NN_INDEX,
               SCAMP::PRECISION_SINGLE, SupportedConfig());
  writer.Save();

  EXPECT_TRUE(std::filesystem::is_regular_file(file));
  std::filesystem::remove_all(
      std::filesystem::temp_directory_path() / "scamp_test_mkdir", ec);
}

// Save() must atomically replace an existing cache file (otherwise a
// stale cache could be served if two processes race). std::rename does
// not replace on Windows; std::filesystem::rename does. Catch the
// regression by overwriting an existing file.
void Test_SaveReplacesExistingFile() {
  SCAMP::ResetAutotuneWarnings();
  std::string path = MakeTempPath("replace");

  // First write: one entry.
  {
    SCAMP::AutotuneCache writer(path);
    SCAMP::KernelConfig cfg = SupportedConfig();
    cfg.blocksz = 64;
    writer.Store("DeviceReplace", SCAMP::PROFILE_TYPE_1NN_INDEX,
                 SCAMP::PRECISION_SINGLE, cfg);
    writer.Save();
  }
  // Second write to the same path: a different entry.
  {
    SCAMP::AutotuneCache writer(path);
    SCAMP::KernelConfig cfg = SupportedConfig();  // blocksz = 128
    writer.Store("DeviceReplace", SCAMP::PROFILE_TYPE_1NN_INDEX,
                 SCAMP::PRECISION_SINGLE, cfg);
    writer.Save();
  }
  // Reader should see the second write.
  SCAMP::AutotuneCache reader(path);
  reader.Load();
  auto found = reader.Lookup("DeviceReplace", SCAMP::PROFILE_TYPE_1NN_INDEX,
                             SCAMP::PRECISION_SINGLE);
  EXPECT_TRUE(found.has_value());
  if (found.has_value()) {
    EXPECT_EQ(found->blocksz, 128);
  }
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(path), ec);
}

// Malformed cache: a non-empty bad line throws SCAMPException. The
// kHeader-mismatch case clears entries silently (existing behavior); we
// don't test that here because it's an intentional escape hatch for
// older formats.
void Test_MalformedLineThrows() {
  bool threw = false;
  try {
    auto bad = MakeCacheFromString(
        "SCAMP_AUTOTUNE_V1\n"
        "DeviceA|1NN_INDEX|SINGLE|128|8|4|0|8\n");  // 8 fields, expected 9
  } catch (const std::exception &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*fn)();
  };
  TestCase cases[] = {
      {"Test_UserCacheHit", Test_UserCacheHit},
      {"Test_BuiltinCacheHit", Test_BuiltinCacheHit},
      {"Test_UserCacheBeatsBuiltin", Test_UserCacheBeatsBuiltin},
      {"Test_CacheMissEmitsWarningAndReturnsFallback",
       Test_CacheMissEmitsWarningAndReturnsFallback},
      {"Test_CacheMissWarningIsOneShot", Test_CacheMissWarningIsOneShot},
      {"Test_QuietEnvSuppressesWarning", Test_QuietEnvSuppressesWarning},
      {"Test_QuietEnv_FalseyValuesStillWarn",
       Test_QuietEnv_FalseyValuesStillWarn},
      {"Test_CacheMissWarningIsPerTuple", Test_CacheMissWarningIsPerTuple},
      {"Test_UnsupportedVariantInCacheFallsThrough",
       Test_UnsupportedVariantInCacheFallsThrough},
      {"Test_MultiDeviceCacheLookup", Test_MultiDeviceCacheLookup},
      {"Test_DefaultPath_EnvVarOverride", Test_DefaultPath_EnvVarOverride},
      {"Test_DefaultPath_XdgCacheHome", Test_DefaultPath_XdgCacheHome},
      {"Test_DefaultPath_UserDirFallback", Test_DefaultPath_UserDirFallback},
      {"Test_DiskRoundTrip", Test_DiskRoundTrip},
      {"Test_SaveCreatesParentDirectories",
       Test_SaveCreatesParentDirectories},
      {"Test_SaveReplacesExistingFile", Test_SaveReplacesExistingFile},
      {"Test_MalformedLineThrows", Test_MalformedLineThrows},
  };
  int n = sizeof(cases) / sizeof(cases[0]);
  for (int i = 0; i < n; ++i) {
    g_current_test = cases[i].name;
    int before = g_failures;
    cases[i].fn();
    if (g_failures == before) {
      std::cout << "[ OK ] " << cases[i].name << "\n";
    }
  }
  std::cout << "\n";
  if (g_failures == 0) {
    std::cout << "All " << n << " test(s) passed.\n";
    return 0;
  }
  std::cerr << g_failures << " failure(s).\n";
  return 1;
}
