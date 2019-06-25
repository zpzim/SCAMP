#include "common.h"
#include "scamp_exception.h"

#include <cstdlib>
#include <sstream>

void SCAMP::SCAMPArgs::validate() {
  if (window < 3) {
    throw SCAMPException("Error: Subsequence length must be at least 3");
  }
  if (max_tile_size < 1024) {
    throw SCAMPException("Error: max tile size must be at least 1024");
  }
  if (max_tile_size / 2 < window) {
    throw SCAMPException(
        "Error: Tile length and width must be at least 2x larger than the "
        "window size");
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    throw SCAMPException(
        "Error: Input time series must be at least 'subesequence window size' "
        "in length");
  }
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
      throw SCAMPException(
          "Error: Could not determine size of profile elements");
  }
}

#ifdef _HAS_CUDA_
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    std::ostringstream ostream;
    ostream << "GPUasssert: " << cudaGetErrorString(code) << " " << file << " "
            << line;
    throw SCAMPException(ostream.str());
  }
}
#endif
