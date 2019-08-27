#include "common.h"
#include "scamp_exception.h"

#include <cstdlib>
#include <sstream>

namespace SCAMP {

void SCAMPArgs::validate() {
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

std::string GetProfileTypeString(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_INVALID:
      return "PROFILE_TYPE_INVALID";
    case PROFILE_TYPE_1NN_INDEX:
      return "PROFILE_TYPE_1NN_INDEX";
    case PROFILE_TYPE_1NN:
      return "PROFILE_TYPE_1NN";
    case PROFILE_TYPE_SUM_THRESH:
      return "PROFILE_TYPE_SUM_THRESH";
    case PROFILE_TYPE_FREQUENCY_THRESH:
      return "PROFILE_TYPE_FREQUENCY_THRESH";
    case PROFILE_TYPE_KNN:
      return "PROFILE_TYPE_KNN";
    case PROFILE_TYPE_1NN_MULTIDIM:
      return "PROFILE_TYPE_1NN_MULTIDIM";
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return "PROFILE_TYPE_APPROX_ALL_NEIGHBORS";
  }
}

std::string GetPrecisionTypeString(SCAMPPrecisionType t) {
  switch (t) {
    case PRECISION_INVALID:
      return "PRECISION_INVALID";
    case PRECISION_SINGLE:
      return "PRECISION_SINGLE";
    case PRECISION_MIXED:
      return "PRECISION_MIXED";
    case PRECISION_DOUBLE:
      return "PRECISION_DOUBLE";
  }
}

std::string getSCAMPErrorString(SCAMPError_t err) {
  switch (err) {
    case SCAMP_NO_ERROR:
      return "SCAMP_NO_ERROR";
    case SCAMP_FUNCTIONALITY_UNIMPLEMENTED:
      return "SCAMP_FUNCTIONALITY_UNIMPLEMENTED";
    case SCAMP_TILE_ILLEGAL_TYPE:
      return "SCAMP_TILE_ILLEGAL_TYPE";
    case SCAMP_CUDA_ERROR:
      return "SCAMP_CUDA_ERROR";
    case SCAMP_CUFFT_ERROR:
      return "SCAMP_CUFFT_ERROR";
    case SCAMP_CUFFT_EXEC_ERROR:
      return "SCAMP_CUFFT_EXEC_ERROR";
    case SCAMP_DIM_INCOMPATIBLE:
      return "SCAMP_DIM_INCOMPATIBLE";
  }
}

void SCAMPArgs::print() {
  std::cout << "window: " << window << std::endl;
  std::cout << "max_tile_size: " << max_tile_size << std::endl;
  std::cout << "has_b: " << has_b << std::endl;
  std::cout << "keep_rows_separate: " << keep_rows_separate << std::endl;
  std::cout << "distributed_start_row: " << distributed_start_row << std::endl;
  std::cout << "distributed_start_col: " << distributed_start_col << std::endl;
  std::cout << "computing_rows: " << computing_rows << std::endl;
  std::cout << "computing_columns: " << computing_columns << std::endl;
  std::cout << "is_aligned: " << is_aligned << std::endl;
  std::cout << "profile_type: " << GetProfileTypeString(profile_type)
            << std::endl;
  std::cout << "precision_type: " << GetPrecisionTypeString(precision_type)
            << std::endl;
  std::cout << "distance_threshold: " << distance_threshold << std::endl;
  std::cout << "silent_mode: " << silent_mode << std::endl;
  std::cout << "max_matches_per_column: " << max_matches_per_column
            << std::endl;
  std::cout << "timeseries_a size: " << timeseries_a.size() << std::endl;
  std::cout << "timeseries_b size: " << timeseries_b.size() << std::endl;
}

size_t GetProfileTypeSize(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      return sizeof(double);
    case PROFILE_TYPE_1NN_INDEX:
      return sizeof(uint64_t);
    case PROFILE_TYPE_1NN:
      return sizeof(float);
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return sizeof(SCAMPmatch);
    default:
      throw SCAMPException(
          "Error: Could not determine size of profile elements");
  }
}

}  // namespace SCAMP

#ifdef _HAS_CUDA_
void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::ostringstream ostream;
    ostream << "GPUasssert: " << cudaGetErrorString(code) << " " << file << " "
            << line;
    throw SCAMPException(ostream.str());
  }
}
#endif
