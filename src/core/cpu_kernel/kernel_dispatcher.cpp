#include "kernel_dispatcher.h"

#ifdef _SCAMP_DISTRIBUTABLE_
#include "avx/dispatch_avx.h"
#include "avx2/dispatch_avx2.h"
#endif
#include "baseline/dispatch_baseline.h"

#if defined(_SCAMP_DISTRIBUTABLE_)
#include "cpu_features_macros.h"
#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
static const cpu_features::X86Features features =
    cpu_features::GetX86Info().features;
#endif
#endif

namespace SCAMP {

SCAMPError_t dispatch(SCAMPKernelInputArgs<double> args, Tile *t,
                      void *profile_a, void *profile_b, bool do_rows,
                      bool do_cols) {
#ifndef _SCAMP_DISTRIBUTABLE_
  if (!t->info()->silent_mode) {
    std::cout
        << "Launching kernel optimized for the platform SCAMP was built on."
        << std::endl;
  }
  return dispatch_kernel_baseline(args, t, profile_a, profile_b, do_rows,
                                  do_cols);
#elif defined(CPU_FEATURES_ARCH_X86)
  if (features.avx && features.avx2 && features.fma3) {
    if (!t->info()->silent_mode) {
      std::cout << "Launching AVX2 kernel." << std::endl;
    }
    return dispatch_kernel_avx2(args, t, profile_a, profile_b, do_rows,
                                do_cols);
  }
  if (features.avx) {
    if (!t->info()->silent_mode) {
      std::cout << "Launching AVX kernel." << std::endl;
    }
    return dispatch_kernel_avx(args, t, profile_a, profile_b, do_rows, do_cols);
  }
#endif
  if (!t->info()->silent_mode) {
    std::cout << "Launching baseline kernel." << std::endl;
  }
  return dispatch_kernel_baseline(args, t, profile_a, profile_b, do_rows,
                                  do_cols);
}

SCAMPError_t cpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, false);
  return dispatch(tile_args, t, t->profile_a(), t->profile_b(),
                  t->info()->computing_rows, t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, false);
  return dispatch(tile_args, t, t->profile_b(), t->profile_a(),
                  t->info()->computing_cols, t->info()->computing_rows);
}

SCAMPError_t cpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, true);
  return dispatch(tile_args, t, t->profile_a(), t->profile_b(),
                  t->info()->computing_rows, t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, true);
  return dispatch(tile_args, t, t->profile_b(), t->profile_a(),
                  t->info()->computing_cols, t->info()->computing_rows);
}

}  // namespace SCAMP
