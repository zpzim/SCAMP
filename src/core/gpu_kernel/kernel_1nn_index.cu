// Per-profile launch dispatcher for PROFILE_TYPE_1NN_INDEX. The heavy
// do_tile instantiations live in kernel_1nn_index_v0.cu ...
// kernel_1nn_index_v5.cu (generated from kernel_variant.cu.in); this file
// resolves cfg -> variant index.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_variants.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_1NN_INDEX(SCAMPKernelInputArgs<double> args,
                                    uint64_t *profile_A, uint64_t *profile_B,
                                    SCAMPPrecisionType fp_type,
                                    bool computing_rows, bool computing_cols,
                                    KernelConfig cfg, uint64_t num_blocks,
                                    uint64_t smem, cudaStream_t s) {
  SCAMP_VARIANT_DISPATCH(1NN_INDEX);
}

}  // namespace SCAMP
