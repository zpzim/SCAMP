// Per-profile launch dispatcher for PROFILE_TYPE_MATRIX_SUMMARY. The heavy
// do_tile instantiations live in kernel_matrix_summary_v0.cu ...
// kernel_matrix_summary_v5.cu (generated from kernel_variant.cu.in); this
// file resolves cfg -> variant index.
#include "kernel_gpu_utils.h"
#include "kernels_dispatch.h"
#include "kernels_variants.h"

namespace SCAMP {

SCAMPError_t LaunchKernel_MATRIX_SUMMARY(SCAMPKernelInputArgs<double> args,
                                         float *profile_A, float *profile_B,
                                         SCAMPPrecisionType fp_type,
                                         bool computing_rows,
                                         bool computing_cols, KernelConfig cfg,
                                         uint64_t num_blocks, uint64_t smem,
                                         cudaStream_t s) {
  SCAMP_VARIANT_DISPATCH(MATRIX_SUMMARY);
}

}  // namespace SCAMP
