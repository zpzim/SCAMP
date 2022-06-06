#include "dispatch_baseline.h"

#include "core/cpu_kernel/cpu_kernels.h"

namespace SCAMP {

SCAMPError_t dispatch_kernel_baseline(SCAMPKernelInputArgs<double> args,
                                      Tile *t, void *profile_a, void *profile_b,
                                      bool do_rows, bool do_cols) {
  return BASELINE::compute_cpu_resources_and_launch(
      args, t, profile_a, profile_b, do_rows, do_cols);
}

}  // namespace SCAMP
