#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cinttypes>
#include <unordered_map>
#include "SCAMP.pb.h"
namespace SCAMP {

typedef union {
  float floats[2];       // floats[0] = lowest
  unsigned int ints[2];  // ints[1] = lowIdx
  uint64_t ulong;        // for atomic update
} mp_entry;

template <unsigned int count>
struct reg_mem {
  float dist[count];
  double qt[count];
};

struct OptionalArgs {
  OptionalArgs(double threshold_) : threshold(threshold_) {}

  double threshold;
};

struct OpInfo {
  OpInfo(size_t Asize, size_t Bsize, size_t window_sz, size_t max_tile_size,
         bool selfjoin, SCAMPPrecisionType t, int64_t start_row,
         int64_t start_col, OptionalArgs args_, SCAMPProfileType profiletype,
         bool keep_rows, bool compute_rows, bool compute_cols, bool aligned,
         int num_workers)
      : full_ts_len_A(Asize),
        full_ts_len_B(Bsize),
        mp_window(window_sz),
        self_join(selfjoin),
        fp_type(t),
        global_start_row_position(start_row),
        global_start_col_position(start_col),
        opt_args(args_),
        profile_type(profiletype),
        keep_rows_separate(keep_rows),
        computing_rows(compute_rows),
        computing_cols(compute_cols),
        is_aligned(aligned) {
    if (self_join) {
      full_ts_len_B = full_ts_len_A;
    }
    max_tile_ts_size = std::max(Asize, Bsize) / (num_workers);
    if (max_tile_ts_size > max_tile_size) {
      max_tile_ts_size = max_tile_size;
    }
    max_tile_width = max_tile_ts_size - mp_window + 1;
    max_tile_height = max_tile_width;
  }
  // Type of profile to compute
  SCAMPProfileType profile_type;

  // Total size of A timeseries
  size_t full_ts_len_A;
  // Total size of B timesereis
  size_t full_ts_len_B;
  // Max size of the timeseries associated with the tile
  size_t max_tile_ts_size;
  // Max width of the distance matrix associated with the tile
  size_t max_tile_width;
  // Max height of the distance matrix associated with the tile
  size_t max_tile_height;

  // Subsequence window length for MP
  size_t mp_window;

  // For distributed joins, the start position of this join in relation to other
  // distributed tiles.
  int64_t global_start_row_position;
  int64_t global_start_col_position;

  // Optional kernel arguments
  OptionalArgs opt_args;

  // Precision type of computation
  SCAMPPrecisionType fp_type;
  // Whether or not we are computing a self join (symmetric distance matrix)
  bool self_join;
  // Whether or not to compute MP along the rows.
  bool computing_rows;
  // Whether or not to compute MP along the columns.
  bool computing_cols;
  // Whether or not time series A and B start with the same prefix.
  bool is_aligned;
  // Determines if we should keep the row/column matrix profiles separate or to
  // merge them.
  bool keep_rows_separate;
};

struct PrecomputedInfo {
 private:
  std::vector<double> _norms;
  std::vector<double> _df;
  std::vector<double> _dg;
  std::vector<double> _means;

 public:
  const std::vector<double>& dg() const { return _dg; }
  const std::vector<double>& df() const { return _df; }
  const std::vector<double>& norms() const { return _norms; }
  const std::vector<double>& means() const { return _means; }
  std::vector<double>& mutable_dg() { return _dg; }
  std::vector<double>& mutable_df() { return _df; }
  std::vector<double>& mutable_norms() { return _norms; }
  std::vector<double>& mutable_means() { return _means; }
};

using DeviceProfile = std::unordered_map<int, void*>;

size_t GetProfileTypeSize(SCAMPProfileType t);

enum SCAMPArchitecture {
  CPU_WORKER,
  CUDA_GPU_WORKER,
};

enum SCAMPError_t {
  SCAMP_NO_ERROR,
  SCAMP_FUNCTIONALITY_UNIMPLEMENTED,
  SCAMP_TILE_ILLEGAL_TYPE,
  SCAMP_CUDA_ERROR,
  SCAMP_CUFFT_ERROR,
  SCAMP_CUFFT_EXEC_ERROR,
  SCAMP_DIM_INCOMPATIBLE
};

enum SCAMPTileType {
  SELF_JOIN_FULL_TILE,
  SELF_JOIN_UPPER_TRIANGULAR,
  AB_JOIN_FULL_TILE,
  AB_FULL_JOIN_FULL_TILE
};

}  // namespace SCAMP

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true);
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
