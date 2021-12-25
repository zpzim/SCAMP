#pragma once

#include "common.h"
#include "profile.h"

#include <vector>

namespace SCAMP {

// Arguments for a SCAMP operation
// This is an external user's interface to the SCAMP library
struct SCAMPArgs {
  void validate();
  void print();
  bool InitProfileMemory();

  std::vector<double> timeseries_a;
  std::vector<double> timeseries_b;
  Profile profile_a;
  Profile profile_b;
  bool has_b;
  uint64_t window;
  uint64_t max_tile_size;
  int64_t distributed_start_row;
  int64_t distributed_start_col;
  double distance_threshold;
  SCAMPPrecisionType precision_type;
  SCAMPProfileType profile_type;
  bool computing_rows;
  bool computing_columns;
  bool keep_rows_separate;
  bool is_aligned;
  bool silent_mode;
  int64_t max_matches_per_column;
  int64_t matrix_height;
  int64_t matrix_width;
};

}  // namespace SCAMP
