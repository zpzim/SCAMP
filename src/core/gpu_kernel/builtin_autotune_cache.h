// String constant containing the contents of data/autotune_cache.txt as it
// existed at build time. The CMake configuration step reads the on-disk file
// and embeds it via configure_file into builtin_autotune_cache.cpp.
//
// Lookup order at runtime (see autotune.cpp::GetKernelConfigForDevice):
//   1. User override file ($SCAMP_AUTOTUNE_CACHE, $XDG_CACHE_HOME, then
//      $HOME on Linux/macOS or %LOCALAPPDATA% on Windows; see
//      AutotuneCache::DefaultPath)
//   2. This built-in cache
//   3. GetDefaultKernelConfig() as a last resort
#pragma once

namespace SCAMP {

extern const char *kBuiltinAutotuneCache;

}  // namespace SCAMP
