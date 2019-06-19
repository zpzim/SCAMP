#include "common.h"

#include <cstdlib>

bool SCAMP::SCAMPArgs::valid() {
  if (window < 3) {
    printf("Error: Subsequence length must be at least 3\n");
    return false;
  }
  if (max_tile_size < 1034) {
    printf("Error: max tile size must be at least 1024\n");
    return false;
  }
  if (max_tile_size / 2 < window) {
    printf(
        "Error: Tile length and width must be at least 2x larger than the "
        "window size\n");
    return false;
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    printf(
        "Error: Input time series must be at least 'subesequence window size' "
        "in length\n");
    return false;
  }
  return true;
}

size_t SCAMP::GetProfileTypeSize(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      return sizeof(double);
    case PROFILE_TYPE_1NN_INDEX:
      return sizeof(uint64_t);
    case PROFILE_TYPE_1NN:
      return sizeof(float);
    default:
      printf("Error: Could not determine size of profile elements");
      exit(1);
      return 0;
  }
}

#ifdef _HAS_CUDA_
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      exit(code);
    }
  }
}
#endif
