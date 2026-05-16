// Instantiates the SCAMP do_tile<...> kernel for PROFILE_TYPE_SUM_THRESH.
// Split out of kernels.cu so the heavy template instantiations for each
// profile type can compile in parallel.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_impl.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_SUM_THRESH(SCAMPKernelInputArgs<double> args,
                                     double *profile_A, double *profile_B,
                                     SCAMPPrecisionType fp_type,
                                     bool computing_rows, bool computing_cols,
                                     KernelConfig cfg, uint64_t num_blocks,
                                     uint64_t smem, cudaStream_t s) {
  return LaunchDoTile<double, double, double, PROFILE_TYPE_SUM_THRESH>(
      args, profile_A, profile_B, fp_type, computing_rows, computing_cols, cfg,
      num_blocks, smem, s);
}

}  // namespace SCAMP
