#include <cuda_runtime.h>
#include "kernel_gpu_utils.h"

namespace SCAMP {

int get_blocksz(Tile *t) {
  switch (t->info()->fp_type) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return BLOCKSZ_DP;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return BLOCKSZ_SP;
  }
  return 0;
}

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

int GetTileHeight(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_ULTRA:
    case PRECISION_DOUBLE:
      return TILE_HEIGHT_DP;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return TILE_HEIGHT_SP;
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

int get_smem(const OpInfo *info, uint64_t blocksz) {
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(info->fp_type);
  int tile_height = GetTileHeight(info->fp_type);
  int tile_width = blocksz * DIAGS_PER_THREAD + tile_height;
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

}  // namespace SCAMP
