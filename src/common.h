#pragma once

#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#endif

#include <stdio.h>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <unordered_map>

namespace SCAMP {

// Types of matrix profile to compute
enum SCAMPProfileType {
  PROFILE_TYPE_INVALID = 0,
  PROFILE_TYPE_1NN_INDEX = 1,
  PROFILE_TYPE_SUM_THRESH = 2,
  PROFILE_TYPE_FREQUENCY_THRESH = 3,
  PROFILE_TYPE_KNN = 4,
  PROFILE_TYPE_1NN_MULTIDIM = 5,
  PROFILE_TYPE_1NN = 6,
};

// Precision modes
enum SCAMPPrecisionType {
  PRECISION_INVALID = 0,
  PRECISION_SINGLE = 1,
  PRECISION_MIXED = 2,
  PRECISION_DOUBLE = 3,
};

// For computing the 1NN Matrix profile and index on the GPU, we store both the
// index and distance as a single 64 bit value which allows for atomic updating
// on the GPU
typedef union {
  float floats[2];       // floats[0] = lowest
  unsigned int ints[2];  // ints[1] = lowIdx
  uint64_t ulong;        // for atomic update
} mp_entry;

struct ProfileData {
  // Only one of these should be set at once
  std::vector<uint32_t> uint32_value;
  std::vector<uint64_t> uint64_value;
  std::vector<float> float_value;
  std::vector<double> double_value;
};

// Stores information about a matrix profile
struct Profile {
  std::vector<ProfileData> data;
  SCAMPProfileType type;
};

// Arguments for a SCAMP operation
// This is an external user's interface to the SCAMP library
struct SCAMPArgs {
  void validate();

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
  bool left_right;
};

// Struct describing kernel arguments which are non-standard
struct OptionalArgs {
  OptionalArgs() : threshold(NAN), num_extra_operands(0) {}
  OptionalArgs(double threshold_)
      : threshold(threshold_), num_extra_operands(0) {}
  OptionalArgs(double threshold_, int num_extra_operands_)
      : threshold(threshold_), num_extra_operands(num_extra_operands_) {}

  int num_extra_operands;
  double threshold;
};

// Struct defines information about a SCAMP Operation
struct OpInfo {
  OpInfo(size_t Asize, size_t Bsize, size_t window_sz, size_t max_tile_size,
         bool selfjoin, SCAMPPrecisionType t, int64_t start_row,
         int64_t start_col, OptionalArgs args_, SCAMPProfileType profiletype,
         bool keep_rows, bool compute_rows, bool compute_cols, bool aligned,
         bool silent_mode, bool leftright, int num_workers)
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
        is_aligned(aligned),
        silent_mode(silent_mode),
        left_right(leftright) {
    if (self_join) {
      full_ts_len_B = full_ts_len_A;
    }
    auto maxSize = std::max(Asize, Bsize);
    max_tile_ts_size = maxSize / (num_workers);

    if (max_tile_ts_size > max_tile_size) {
      max_tile_ts_size = max_tile_size;
    } else if (max_tile_ts_size < mp_window) {
      max_tile_ts_size = maxSize;
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
  // Run without printing any message by standard output
  bool silent_mode;

  bool left_right;
};

// Struct containing the precomputed statistics for an input time series
struct PrecomputedInfo {
 private:
  std::vector<double> _norms;
  std::vector<double> _df;
  std::vector<double> _dg;
  std::vector<double> _means;

 public:
  void set(std::vector<double> &means, std::vector<double> &norms,
           std::vector<double> &df, std::vector<double> &dg) {
    _norms = std::move(norms);
    _means = std::move(means);
    _df = std::move(df);
    _dg = std::move(dg);
  }

  const std::vector<double> &dg() const { return _dg; }
  const std::vector<double> &df() const { return _df; }
  const std::vector<double> &norms() const { return _norms; }
  const std::vector<double> &means() const { return _means; }
  std::vector<double> &mutable_dg() { return _dg; }
  std::vector<double> &mutable_df() { return _df; }
  std::vector<double> &mutable_norms() { return _norms; }
  std::vector<double> &mutable_means() { return _means; }
};

// Thread safe queue to hold tiles to be executed
class ThreadSafeQueue {
 public:
  size_t size() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.size();
  }

  bool empty() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.empty();
  }

  // Pop an element from the queue, if the queue is already empty, return the
  // sentinel (-1,-1) which indicates that there was no data in the queue
  std::pair<int, int> pop() {
    std::unique_lock<std::mutex> mlock(mutex_);
    auto item = std::pair<int, int>(-1, -1);
    if (!queue_.empty()) {
      item = queue_.front();
      queue_.pop();
    }
    return item;
  }

  void push(const std::pair<int, int> &item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void push(std::pair<int, int> &&item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(std::move(item));
    mlock.unlock();
    cond_.notify_one();
  }

 private:
  std::queue<std::pair<int, int>> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

using DeviceProfile = std::unordered_map<int, void *>;

// Returns the size in bytes of a the matrix profile element for a particular MP
// type
size_t GetProfileTypeSize(SCAMPProfileType t);

// Enum describing worker architecture, used to switch on architecture specific
// code
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

#ifdef _HAS_CUDA_
void gpuAssert(cudaError_t code, const char *file, int line);
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
#endif

#define ASSERT(condition, message)                                         \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::ostringstream ostream;                                          \
      ostream << "Assertion `" << #condition << "` failed in " << __FILE__ \
              << "line " << __LINE__;                                      \
      throw SCAMPException(ostream.str());                                 \
    }                                                                      \
  } while (false)
