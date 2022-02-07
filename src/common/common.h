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
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "scamp_exception.h"

#ifdef _HAS_CUDA_
#define HOST_DEVICE_FUNCTION __host__ __device__
#else
#define HOST_DEVICE_FUNCTION
#endif

namespace SCAMP {

using DeviceProfile = std::unordered_map<int, void *>;

struct OpInfo;
struct ExecInfo;

// Types of matrix profile to compute
enum SCAMPProfileType {
  PROFILE_TYPE_INVALID = 0,
  PROFILE_TYPE_1NN_INDEX = 1,
  PROFILE_TYPE_SUM_THRESH = 2,
  PROFILE_TYPE_FREQUENCY_THRESH = 3,  // Unused
  PROFILE_TYPE_KNN = 4,               // Unused
  PROFILE_TYPE_1NN_MULTIDIM = 5,      // Unused
  PROFILE_TYPE_1NN = 6,
  PROFILE_TYPE_APPROX_ALL_NEIGHBORS = 7,
  PROFILE_TYPE_MATRIX_SUMMARY = 8,
};

// Precision modes
enum SCAMPPrecisionType {
  PRECISION_INVALID = 0,
  PRECISION_SINGLE = 1,
  PRECISION_MIXED = 2,
  PRECISION_DOUBLE = 3,
  PRECISION_ULTRA = 4,
};

std::string GetPrecisionTypeString(SCAMPPrecisionType t);
std::string GetProfileTypeString(SCAMPProfileType t);
bool NeedsSort(SCAMPProfileType type);
bool NeedsIntermittentMerge(SCAMPProfileType type);
bool NeedsIntermittentReset(SCAMPProfileType type);

// Enum describing worker architecture, used to switch on architecture specific
// code
enum SCAMPArchitecture {
  CPU_WORKER,
  CUDA_GPU_WORKER,
};

// For computing the 1NN Matrix profile and index on the GPU, we store both the
// index and distance as a single 64 bit value which allows for atomic updating
// on the GPU
typedef union {
  float floats[2];       // floats[0] = lowest
  unsigned int ints[2];  // ints[1] = lowIdx
  uint64_t ulong;        // for atomic update
} mp_entry;

struct SCAMPmatch {
  HOST_DEVICE_FUNCTION SCAMPmatch() : corr(-2), row(0), col(0) {}
  HOST_DEVICE_FUNCTION SCAMPmatch(float d, uint32_t r, uint32_t c)
      : corr(d), row(r), col(c) {}
  HOST_DEVICE_FUNCTION bool operator<(const SCAMPmatch &other) const {
    if (col == other.col) {
      return corr > other.corr;
    }
    return col < other.col;
  }
  float corr;
  uint32_t row;
  uint32_t col;
};

class compareMatch {
 public:
  bool operator()(const SCAMPmatch &x1, const SCAMPmatch &x2) {
    return x1.corr > x2.corr;
  }
};

void Memcopy(void *destination, const void *source, size_t bytes,
             bool from_tile, const ExecInfo *info);

// Struct describing kernel arguments which are non-standard
struct OptionalArgs {
  OptionalArgs() : threshold(NAN), num_extra_operands(0) {}
  OptionalArgs(double threshold_)
      : threshold(threshold_), num_extra_operands(0) {}
  OptionalArgs(double threshold_, int num_extra_operands_)
      : threshold(threshold_), num_extra_operands(num_extra_operands_) {}

  double threshold;
  int num_extra_operands;
};

// Defines the execution environment of a SCAMP tile
struct ExecInfo {
  SCAMPArchitecture arch;
  int cuda_id;
#ifdef _HAS_CUDA_
  cudaStream_t stream;
  cudaDeviceProp dev_props;
#endif
  ExecInfo(SCAMPArchitecture _arch, int _cuda_id);
  ~ExecInfo();
};

// Struct defines information about a SCAMP Operation
struct OpInfo {
  OpInfo(size_t Asize, size_t Bsize, size_t window_sz, size_t max_tile_size,
         bool selfjoin, SCAMPPrecisionType t, int64_t start_row,
         int64_t start_col, OptionalArgs args_, SCAMPProfileType profiletype,
         bool keep_rows, bool compute_rows, bool compute_cols, bool aligned,
         bool silent_mode, int num_workers, int64_t max_matches_per_col,
         int64_t mheight, int64_t mwidth);

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

  // Type of profile to compute
  SCAMPProfileType profile_type;

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
  // Max matches per column for ALL_NEIGHBORS profile type
  int64_t max_matches_per_column;
  // Max matches per tile for ALL_NEIGHBORS profile type
  int64_t max_matches_per_tile;
  // Variables associated with the MATRIX_SUMMARY profile type
  int64_t matrix_height;
  int64_t matrix_width;
  double cols_per_cell;
  double rows_per_cell;
};

// Struct containing the precomputed statistics for an input time series
struct PrecomputedInfo {
 private:
  std::vector<double> norms_;
  std::vector<double> df_;
  std::vector<double> dg_;
  std::vector<double> means_;
  std::vector<int> nan_idxs_;

 public:
  void set(std::vector<double> &means, std::vector<double> &norms,
           std::vector<double> &df, std::vector<double> &dg,
           std::vector<int> &nan_idxs) {
    norms_ = std::move(norms);
    means_ = std::move(means);
    df_ = std::move(df);
    dg_ = std::move(dg);
    nan_idxs_ = std::move(nan_idxs);
  }

  const std::vector<double> &dg() const { return dg_; }
  const std::vector<double> &df() const { return df_; }
  const std::vector<double> &norms() const { return norms_; }
  const std::vector<double> &means() const { return means_; }
  const std::vector<int> &nan_idxs() const { return nan_idxs_; }
  std::vector<double> &mutable_dg() { return dg_; }
  std::vector<double> &mutable_df() { return df_; }
  std::vector<double> &mutable_norms() { return norms_; }
  std::vector<double> &mutable_means() { return means_; }
};

struct CombinedStats {
 public:
  std::vector<double> dr_fwd;
  std::vector<double> dr_bkwd;
  std::vector<double> dc_fwd;
  std::vector<double> dc_bkwd;
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

// Returns the size in bytes of a the matrix profile element for a particular MP
// type
size_t GetProfileTypeSize(SCAMPProfileType t);

// Enum describing different types of SCAMP errors
enum SCAMPError_t {
  SCAMP_NO_ERROR,
  SCAMP_FUNCTIONALITY_UNIMPLEMENTED,
  SCAMP_TILE_ILLEGAL_TYPE,
  SCAMP_CUDA_ERROR,
  SCAMP_CUFFT_ERROR,
  SCAMP_CUFFT_EXEC_ERROR,
  SCAMP_DIM_INCOMPATIBLE
};

// Returns the string associated with a SCAMPError_t
std::string getSCAMPErrorString(SCAMPError_t err);

// Enum describing different tile execution configurations for SCAMP
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
