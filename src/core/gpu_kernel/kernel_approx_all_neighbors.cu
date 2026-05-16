// Instantiates the SCAMP do_tile<...> kernel for
// PROFILE_TYPE_APPROX_ALL_NEIGHBORS. Split out of kernels.cu so the heavy
// template instantiations for each profile type can compile in parallel.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_impl.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_APPROX_ALL_NEIGHBORS(
    SCAMPKernelInputArgs<double> args, SCAMPmatch *profile_A,
    SCAMPmatch *profile_B, SCAMPPrecisionType fp_type, bool computing_rows,
    bool computing_cols, KernelConfig cfg, uint64_t num_blocks, uint64_t smem,
    cudaStream_t s) {
  return LaunchDoTile<SCAMPmatch, uint64_t, float,
                      PROFILE_TYPE_APPROX_ALL_NEIGHBORS>(
      args, profile_A, profile_B, fp_type, computing_rows, computing_cols, cfg,
      num_blocks, smem, s);
}

}  // namespace SCAMP
