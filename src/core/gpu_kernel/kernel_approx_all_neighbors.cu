// Per-profile launch dispatcher for PROFILE_TYPE_APPROX_ALL_NEIGHBORS. The
// heavy do_tile instantiations live in kernel_approx_all_neighbors_v0.cu
// ... kernel_approx_all_neighbors_v5.cu (generated from
// kernel_variant.cu.in); this file resolves cfg -> variant index.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_variants.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_APPROX_ALL_NEIGHBORS(
    SCAMPKernelInputArgs<double> args, SCAMPmatch *profile_A,
    SCAMPmatch *profile_B, SCAMPPrecisionType fp_type, bool computing_rows,
    bool computing_cols, KernelConfig cfg, uint64_t num_blocks, uint64_t smem,
    cudaStream_t s) {
  SCAMP_VARIANT_DISPATCH(APPROX_ALL_NEIGHBORS);
}

}  // namespace SCAMP
