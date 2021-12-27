#include <chrono>
#include <cinttypes>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "SCAMP.h"
#include "common/common.h"
#include "common/scamp_exception.h"
#include "cpu_stats.h"
#include "tile.h"

using std::vector;

namespace SCAMP {

// This method computes all the work that must be done to perform the
// reqiested operation and sets the work order by populating the work
// queue.
void SCAMP_Operation::get_tiles() {
  size_t num_tile_rows = ceil((info_.full_ts_len_B - info_.mp_window + 1) /
                              static_cast<double>(info_.max_tile_height));
  size_t num_tile_cols = ceil((info_.full_ts_len_A - info_.mp_window + 1) /
                              static_cast<double>(info_.max_tile_width));
  if (!info_.silent_mode) {
    printf("num_tile_rows = %lu, cols = %lu\n", num_tile_rows, num_tile_cols);
  }

  if (info_.self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        work_queue_.push(std::make_pair(diag, diag + offset));
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      work_queue_.push(std::make_pair(i, num_tile_cols - 1));
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        work_queue_.push(std::make_pair(offset, diag + offset));
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        work_queue_.push(std::make_pair(offset + diag, offset));
      }
    }

    // Add the corner edge tile
    work_queue_.push(std::make_pair(num_tile_rows - 1, num_tile_cols - 1));

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      work_queue_.push(std::make_pair(y, num_tile_cols - 1));
      work_queue_.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      work_queue_.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      work_queue_.push(std::make_pair(y, num_tile_cols - 1));
      ++y;
    }
  }
  total_tiles_ = work_queue_.size();
}

// This function is the point of entry for all worker threads
// Workers will initialize a memoty space to work with (Tile)
// and compute the partial matrix profile corresponding with
// the tile. Afterwards, the worker will merge its partial
// solution with the global matrix profile
void SCAMP_Operation::do_work(const std::vector<double> &timeseries_a,
                              const std::vector<double> &timeseries_b,
                              const OpInfo *info, const SCAMPArchitecture arch,
                              const int device_id) {
  // Init working memory and op/tile specific variables
  Tile tile(info, arch, device_id);
  if (!NeedsIntermittentReset(info_.profile_type)) {
    tile.InitProfile(profile_a_, profile_b_);
  }
  while (!work_queue_.empty()) {
    std::pair<int, int> t = work_queue_.pop();
    if (t.first == -1 && t.second == -1) {
      // Another thread grabbed our tile and now the queue is empty
      break;
    }
    // Get the position of the tile we will compute
    tile.set_tile_row(t.first * info_.max_tile_height);
    tile.set_tile_col(t.second * info_.max_tile_width);
    // Get the size of the tile we will compute
    tile.set_tile_width(std::min(info_.max_tile_ts_size,
                                 info_.full_ts_len_A - tile.get_tile_col()));
    tile.set_tile_height(std::min(info_.max_tile_ts_size,
                                  info_.full_ts_len_B - tile.get_tile_row()));
    if (!info_.silent_mode) {
      std::cout << "Starting tile with starting row of " << tile.get_tile_row()
                << " starting column of " << tile.get_tile_col()
                << " with height " << tile.get_tile_height() << " and width "
                << tile.get_tile_width() << std::endl;
    }
    // Copy the portion of the time series and stats
    //  we will be using from the global arrays.
    tile.InitTimeseries(timeseries_a, timeseries_b);
    tile.InitStats(precompA_, precompB_, precomp_);

    bool done = false;
    while (!done) {
      // Copy the portion of the best-so-far profile
      // we will be using.
      if (NeedsIntermittentReset(info_.profile_type)) {
        tile.InitProfile(profile_a_, profile_b_);
      }
      SCAMPError_t err;
      if (info_.self_join) {
        if (t.first == t.second) {
          // Partial tile on diagonal
          err = tile.execute(SELF_JOIN_UPPER_TRIANGULAR);
        } else {
          // Full Tile
          err = tile.execute(SELF_JOIN_FULL_TILE);
        }
      } else {
        // AB-join
        err = tile.execute(AB_FULL_JOIN_FULL_TILE);
      }
      if (err != SCAMP_NO_ERROR) {
        throw SCAMPException("ERROR " + getSCAMPErrorString(err) +
                             " executing tile");
      }
      // Merge join result
      if (NeedsIntermittentMerge(info_.profile_type)) {
        done = tile.MergeProfile(profile_a_, profile_b_);
      } else {
        done = true;
      }
    }
    // Update our counter with a lock
    std::unique_lock<std::mutex> lock(counter_lock_);
    completed_tiles_++;
  }
  if (!NeedsIntermittentMerge(info_.profile_type)) {
    tile.MergeProfile(profile_a_, profile_b_);
  }
}

// This method is the top level interface for a user to request
// that a join between timeseries_a and timeseries_b be computied
// using the configuration set up in SCAMP_Operation's constructor
SCAMPError_t SCAMP_Operation::do_join(const std::vector<double> &timeseries_a,
                                      const std::vector<double> &timeseries_b) {
  const int num_workers = cpu_workers_ + devices_.size();
  if (!info_.silent_mode) {
    printf("Num workers = %d\n", num_workers);
  }

  std::vector<double> timeseries_a_clean, timeseries_b_clean;
  std::vector<bool> nanvals_a, nanvals_b;

  if (!info_.silent_mode) {
    std::cout << "Precomputing statisics on the CPU." << std::endl;
  }

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  // Remove NaN/inf values and replace with 0, this allows us to tolerate these
  // values during the calculation. nanvals contains the subsequences which
  // contain non-finite values, we use these to force distance calculations
  // for these subsequences to result in NaN.
  convert_non_finite_to_zero(timeseries_a, info_.mp_window, &timeseries_a_clean,
                             &nanvals_a);
  convert_non_finite_to_zero(timeseries_b, info_.mp_window, &timeseries_b_clean,
                             &nanvals_b);

  bool ultra_precision = info_.fp_type == PRECISION_ULTRA;

  // Compute statistics for entire problem
  compute_statistics_cpu(timeseries_a_clean, nanvals_a, &precompA_,
                         info_.mp_window, ultra_precision);
  compute_statistics_cpu(timeseries_b_clean, nanvals_b, &precompB_,
                         info_.mp_window, ultra_precision);

  if (ultra_precision) {
    precomp_ = compute_combined_stats_cpu(timeseries_a_clean, precompA_.means(),
                                          timeseries_b_clean, info_.mp_window,
                                          ultra_precision);
  }

  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();

  if (!info_.silent_mode) {
    double precomputes_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        static_cast<double>(1000000);
    std::cout << "Precomputation took " << precomputes_time << " seconds."
              << std::endl;
  }

  // Populate Work Queue with tiles
  get_tiles();

  if (!info_.silent_mode) {
    std::cout << "Performing join with " << work_queue_.size() << " tiles."
              << std::endl;
  }
  std::vector<std::future<void>> futures(num_workers);

  if (!info_.silent_mode) {
    std::cout << "Main SCAMP thread spawning worker threads." << std::endl;
  }

  // Start CUDA Workers
  for (int i = 0; i < devices_.size(); ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this,
                            timeseries_a_clean, timeseries_b_clean, &info_,
                            CUDA_GPU_WORKER, devices_.at(i));
  }

  // Start CPU Workers
  for (int i = devices_.size(); i < num_workers; ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this,
                            timeseries_a_clean, timeseries_b_clean, &info_,
                            CPU_WORKER, -1);
  }

  // wait for workers to be done
  for (auto &future : futures) {
    future.get();
  }

  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
