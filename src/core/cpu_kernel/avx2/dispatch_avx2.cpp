#include "dispatch_avx2.h"

#include "core/cpu_kernel/cpu_kernels.h"

namespace SCAMP {

SCAMPError_t dispatch_kernel_avx2(SCAMPKernelInputArgs<double> args, Tile *t,
                                  void *profile_a, void *profile_b,
                                  bool do_rows, bool do_cols) {
  return AVX2::compute_cpu_resources_and_launch(args, t, profile_a, profile_b,
                                                do_rows, do_cols);
}

}  // namespace SCAMP
