// String constant containing the contents of data/autotune_cache.txt as it
// existed at build time. The CMake configuration step reads the on-disk file
// and embeds it via configure_file into builtin_autotune_cache.cpp.
//
// Lookup order at runtime (see autotune.cpp::GetKernelConfigForDevice):
//   1. User override file ($SCAMP_AUTOTUNE_CACHE, $XDG_CACHE_HOME, $HOME)
//   2. This built-in cache
//   3. GetDefaultKernelConfig() as a last resort
//
// Conda-forge / pip wheel users have no way to recompile a different binary;
// for them this built-in cache is the only way they get GPU-specific tuning.
// Developers update the on-disk data/autotune_cache.txt and open a PR; the
// next release ships those entries to end users.
#pragma once

namespace SCAMP {

extern const char *kBuiltinAutotuneCache;

}  // namespace SCAMP
