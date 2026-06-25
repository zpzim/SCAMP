#include "common/cuda_to_hip.h"

#if !defined(USE_HIP)
#include <cuda_runtime.h>
#endif

#include "kernel_gpu_utils.h"

namespace SCAMP {

int FPTypeSize(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return sizeof(double);
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return sizeof(float);
    case PRECISION_INVALID:
      return -1;
  }
  return -1;
}

size_t GetProfileTypeSizeInternalGPU(SCAMPProfileType type) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      return sizeof(double);
    case PROFILE_TYPE_1NN_INDEX:
      return sizeof(uint64_t);
    case PROFILE_TYPE_1NN:
      return sizeof(float);
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_MATRIX_SUMMARY:
    case PROFILE_TYPE_KNN:
      return sizeof(uint64_t);
    default:
      throw SCAMPException(
          "Error: Could not determine size of profile elements");
  }
}

int get_smem(const OpInfo *info, uint64_t blocksz, int tile_height,
             int diags_per_thread) {
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(info->fp_type);
  int tile_width = blocksz * diags_per_thread + tile_height;
  int smem = (tile_width + tile_height) *
             (num_shared_variables + info->opt_args.num_extra_operands) *
             intermediate_data_size;
  int profile_data_size = GetProfileTypeSizeInternalGPU(info->profile_type);
  if (info->computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (info->computing_rows) {
    smem += tile_height * profile_data_size;
  }
  if (NeedsCheckIfDone(info->profile_type)) {
    smem += 2 * sizeof(uint64_t);
  }
  return smem;
}

int get_smem_shfl(const OpInfo *info, uint64_t blocksz, int tile_height,
                  int diags_per_thread) {
  constexpr int num_shared_variables = 3;  // df_row, dg_row, inorm_row
  int intermediate_data_size = FPTypeSize(info->fp_type);
  // Tile width = parallelogram width. The staggered column-block rotation
  // makes the block touch columns up to (BLOCKSZ*DPT + tile_height - 1)
  // within one tile; smem.local_mp_col is sized accordingly so
  // state.local_col indexing stays in bounds after rotations.
  int tile_width_profile = blocksz * diags_per_thread + tile_height;
  // cov_handoff holds 2 * warps_per_block scalars, and warps_per_block =
  // blocksz / warp_size on the device. SMALLER warp sizes yield MORE warps,
  // so to bound the host-allocated dynamic smem at or above what the device
  // kernel writes for any runtime warp width, size the hand-off region for
  // the smallest warp width (32, wave32 on RDNA/CUDA). Over-allocating by a
  // few slots on wave64 (CDNA) is harmless; under-allocating would let the
  // device write past the dynamic smem region.
  constexpr int kMinWarpSize = 32;
  int warps_per_block = static_cast<int>(blocksz) / kMinWarpSize;

  // Row-data region: 3 (or more with extra operands) * tile_height *
  // sizeof(T). No column-data region.
  int smem = tile_height *
             (num_shared_variables + info->opt_args.num_extra_operands) *
             intermediate_data_size;
  int profile_data_size = GetProfileTypeSizeInternalGPU(info->profile_type);
  if (info->computing_cols) {
    smem += tile_width_profile * profile_data_size;
  }
  if (info->computing_rows) {
    smem += tile_height * profile_data_size;
  }
  // cov_handoff: 2 * warps_per_block * sizeof(T) (double-buffered).
  smem += 2 * warps_per_block * intermediate_data_size;
  if (NeedsCheckIfDone(info->profile_type)) {
    smem += 2 * sizeof(uint64_t);
  }
  return smem;
}

}  // namespace SCAMP
