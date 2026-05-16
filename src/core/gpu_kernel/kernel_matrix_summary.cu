// Instantiates the SCAMP do_tile<...> kernel for PROFILE_TYPE_MATRIX_SUMMARY.
// Split out of kernels.cu so the heavy template instantiations for each
// profile type can compile in parallel.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_impl.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_MATRIX_SUMMARY(
    SCAMPKernelInputArgs<double> args, float *profile_A, float *profile_B,
    SCAMPPrecisionType fp_type, bool computing_rows, bool computing_cols,
    uint64_t blocksz, uint64_t num_blocks, uint64_t smem, cudaStream_t s) {
  return LaunchDoTile<float, uint64_t, float, PROFILE_TYPE_MATRIX_SUMMARY,
                      BLOCKSPERSM>(args, profile_A, profile_B, fp_type,
                                   computing_rows, computing_cols, blocksz,
                                   num_blocks, smem, s);
}

}  // namespace SCAMP
