#pragma once

#include "common/common.h"
#include "tile.h"

namespace SCAMP {

static constexpr int NUM_EXTRA_OPERANDS = 3;

template <typename T>
struct SCAMPKernelInputArgs {
  SCAMPKernelInputArgs(Tile *t, bool transpose, bool ab_join);
  T *__restrict cov;
  const T *__restrict dfa;
  const T *__restrict dfb;
  const T *__restrict dga;
  const T *__restrict dgb;
  const T *__restrict normsa;
  const T *__restrict normsb;
  const float *__restrict thresholds_a;
  const float *__restrict thresholds_b;
  const T *__restrict extras[NUM_EXTRA_OPERANDS];
  unsigned long long int *profile_a_length;
  unsigned long long int *profile_b_length;
  int64_t max_matches_per_tile;
  int32_t n_x;
  int32_t n_y;
  int32_t exclusion_lower;
  int32_t exclusion_upper;
  int32_t matrix_width;
  int32_t matrix_height;
  double rows_per_cell;
  double cols_per_cell;
  int64_t global_start_col;
  int64_t global_start_row;
  bool has_nan_input;

  OptionalArgs opt;
  void Print() const;
};

}  // namespace SCAMP
