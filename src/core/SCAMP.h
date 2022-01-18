#pragma once

#include <list>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/profile.h"

using std::list;
using std::pair;
using std::unordered_map;
using std::vector;

namespace SCAMP {

class SCAMP_Operation {
 private:
  // Operation specific variables like maximum tile size (see common.h)
  const OpInfo info_;

  // Precomputed statistics used by tiles
  PrecomputedInfo precompA_, precompB_;

  // Precomputed statistics computed from both input A and B.
  CombinedStats precomp_;

  // Result vectors
  Profile *profile_a_, *profile_b_;

  // Tile state variables
  // The order to compute the tiles in, set by get_tiles()
  ThreadSafeQueue work_queue_;

  // Lock for counter updates
  std::mutex counter_lock_;

  // The number of completed tiles
  int completed_tiles_;

  // The total number of tiles
  size_t total_tiles_;

  // Cuda devices to compute with
  std::vector<int> devices_;

  // CPU threads to compute with
  int cpu_workers_;

  void get_tiles();

  void do_work(const std::vector<double> &timeseries_a,
               const std::vector<double> &timeseries_b, const OpInfo *info,
               const SCAMPArchitecture arch, const int device_id);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, int64_t start_row, int64_t start_col,
                  OptionalArgs args_, SCAMPProfileType profile_type,
                  Profile *pA, Profile *pB, bool keep_rows, bool compute_rows,
                  bool compute_cols, bool is_aligned, bool silent_mode,
                  int num_threads, int64_t max_matches_per_col,
                  int64_t matrix_height, int64_t matrix_width)
      : info_(Asize, Bsize, window_sz, max_tile_size, selfjoin, t, start_row,
              start_col, args_, profile_type, keep_rows, compute_rows,
              compute_cols, is_aligned, silent_mode, dev.size() + num_threads,
              max_matches_per_col, matrix_height, matrix_width),
        profile_a_(pA),
        profile_b_(pB),
        completed_tiles_(0),
        devices_(dev),
        cpu_workers_(num_threads) {}

  SCAMPError_t do_join(const std::vector<double> &timeseries_a,
                       const std::vector<double> &timeseries_b);

  int get_completed_tiles() { return completed_tiles_; }
};

}  // namespace SCAMP
