#include "scamp_args.h"

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
        "Error: Tile length and width must be at least 2x larger than "
        "the "
        "window size");
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    throw SCAMPException(
        "Error: Input time series must be at least 'subesequence window "
        "size' "
        "in length");
  }
  if (profile_type == PROFILE_TYPE_MATRIX_SUMMARY && matrix_width > 0 &&
      timeseries_a.size() - window + 1 < matrix_width) {
    throw SCAMPException(
        "Error: Output matrix must have smaller dimensions than the input time "
        "series.");
  }

  if (profile_type == PROFILE_TYPE_MATRIX_SUMMARY && matrix_height > 0 &&
      (has_b ? timeseries_b.size() - window + 1 < matrix_height
             : timeseries_a.size() - window + 1 < matrix_height)) {
    throw SCAMPException(
        "Error: Output matrix must have smaller dimensions than the input time "
        "series.");
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

bool SCAMPArgs::InitProfileMemory() {
  int64_t profile_a_size = timeseries_a.size() - window + 1;
  int64_t profile_b_size =
      has_b ? timeseries_b.size() - window + 1 : profile_a_size;
  if (profile_a_size <= 0 || (keep_rows_separate && profile_b_size <= 0)) {
    // Invalid input
    return false;
  }

  profile_a.Alloc(profile_a_size, matrix_height, matrix_width,
                  distance_threshold);

  if (keep_rows_separate) {
    profile_b.Alloc(profile_b_size, matrix_height, matrix_width,
                    distance_threshold);
  }
  return true;
}

}  // namespace SCAMP
