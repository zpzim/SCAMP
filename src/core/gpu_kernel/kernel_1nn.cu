// Per-profile launch dispatcher for PROFILE_TYPE_1NN. The heavy do_tile
// template instantiations are split into one TU per (profile, variant) pair
// (kernel_1nn_v0.cu ... kernel_1nn_v5.cu, generated from
// kernel_variant.cu.in); this file just resolves cfg -> variant index and
// calls the matching LaunchVariant_1NN_vN helper.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_variants.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_1NN(SCAMPKernelInputArgs<double> args,
                              float *profile_A, float *profile_B,
                              SCAMPPrecisionType fp_type, bool computing_rows,
                              bool computing_cols, KernelConfig cfg,
                              uint64_t num_blocks, uint64_t smem,
                              cudaStream_t s) {
  SCAMP_VARIANT_DISPATCH(1NN);
}

}  // namespace SCAMP
