// Instantiates the SCAMP do_tile<...> kernel for PROFILE_TYPE_1NN_INDEX.
// Split out of kernels.cu so the heavy template instantiations for each
// profile type can compile in parallel.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_impl.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_1NN_INDEX(SCAMPKernelInputArgs<double> args,
                                    uint64_t *profile_A, uint64_t *profile_B,
                                    SCAMPPrecisionType fp_type,
                                    bool computing_rows, bool computing_cols,
                                    uint64_t blocksz, uint64_t num_blocks,
                                    uint64_t smem, cudaStream_t s) {
  return LaunchDoTile<uint64_t, uint64_t, float, PROFILE_TYPE_1NN_INDEX,
                      BLOCKSPERSM>(args, profile_A, profile_B, fp_type,
                                   computing_rows, computing_cols, blocksz,
                                   num_blocks, smem, s);
}

}  // namespace SCAMP
