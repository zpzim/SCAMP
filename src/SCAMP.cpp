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
#include "SCAMPWorker.h"
#include "common.h"
#ifdef _HAS_CUDA_
#include "gpu_stats.h"
#else
#include "cpu_stats.h"
#endif
using std::vector;

namespace SCAMP {

SCAMPError_t SCAMP_Operation::init() { return SCAMP_NO_ERROR; }

SCAMPError_t SCAMP_Operation::destroy() { return SCAMP_NO_ERROR; }

SCAMPError_t SCAMP_Operation::do_tile(
    SCAMPTileType t, Worker *worker,
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h) {
  SCAMPError_t err;
  worker->InitTimeseries(Ta_h, Tb_h);
  worker->InitProfile(_profile_a, _profile_b);
  worker->InitStats(_precompA, _precompB);
  err = worker->execute(t);
  return err;
}

void SCAMP_Operation::get_tiles() {
  size_t num_tile_rows = ceil((_info.full_ts_len_B - _info.mp_window + 1) /
                              static_cast<double>(_info.max_tile_height));
  size_t num_tile_cols = ceil((_info.full_ts_len_A - _info.mp_window + 1) /
                              static_cast<double>(_info.max_tile_width));
  printf("num_tile_rows = %lu, cols = %lu\n", num_tile_rows, num_tile_cols);
  if (_info.self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        _work_queue.push(std::make_pair(diag, diag + offset));
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      _work_queue.push(std::make_pair(i, num_tile_cols - 1));
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset, diag + offset));
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset + diag, offset));
      }
    }

    // Add the corner edge tile
    _work_queue.push(std::make_pair(num_tile_rows - 1, num_tile_cols - 1));

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      ++y;
    }
  }
  _total_tiles = _work_queue.size();
}

void SCAMP_Operation::do_work(
    Worker *worker, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  worker->FirstTimeInit();
  while (!_work_queue.empty()) {
    std::pair<int, int> tile = _work_queue.pop();
    if (tile.first == -1 && tile.second == -1) {
      // Another thread grabbed our tile and now the queue is empty
      break;
    }
    // Get the position of the tile we will compute
    worker->set_tile_row(tile.first * _info.max_tile_height);
    worker->set_tile_col(tile.second * _info.max_tile_width);
    // Get the size of the tile we will compute
    worker->set_tile_width(std::min(
        _info.max_tile_ts_size, _info.full_ts_len_A - worker->get_tile_col()));
    worker->set_tile_height(std::min(
        _info.max_tile_ts_size, _info.full_ts_len_B - worker->get_tile_row()));
    std::cout << "Starting tile with starting row of " << worker->get_tile_row()
              << " starting column of " << worker->get_tile_col()
              << " with height " << worker->get_tile_height() << " and width "
              << worker->get_tile_width() << std::endl;
    SCAMPError_t err;
    if (_info.self_join) {
      if (tile.first == tile.second) {
        // partial tile on diagonal
        err = do_tile(SELF_JOIN_UPPER_TRIANGULAR, worker, timeseries_a,
                      timeseries_b);
      } else {
        // full tile
        err = do_tile(SELF_JOIN_FULL_TILE, worker, timeseries_a, timeseries_b);
      }
    } else if (_info.computing_rows) {
      // BiDirectional AB-join
      err = do_tile(AB_FULL_JOIN_FULL_TILE, worker, timeseries_a, timeseries_b);
    } else {
      // Column AB-join
      err = do_tile(AB_JOIN_FULL_TILE, worker, timeseries_a, timeseries_b);
    }
    if (err != SCAMP_NO_ERROR) {
      printf("ERROR %d executing tile. \n", err);
    }
    // Merge join result
    worker->MergeProfile(_profile_a, _profile_a_lock, _profile_b,
                         _profile_b_lock);
    // FIXME: Protect with LOCK
    std::unique_lock<std::mutex> lock(_counter_lock);
    _completed_tiles++;
  }
  worker->Destroy();
}

SCAMPError_t SCAMP_Operation::do_join(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  const int num_workers = _workers.size();
  printf("Num workers = %d\n", num_workers);

  // Compute statistics
#ifdef _HAS_CUDA_
  compute_statistics_gpu(timeseries_a, &_precompA, _info.mp_window);
  compute_statistics_gpu(timeseries_b, &_precompB, _info.mp_window);
#else
  compute_statistics_cpu(timeseries_a, &_precompA, _info.mp_window);
  compute_statistics_cpu(timeseries_b, &_precompB, _info.mp_window);
#endif

  // Populate Work Queue with tiles
  get_tiles();

  std::cout << "Performing join with " << _work_queue.size() << " tiles."
            << std::endl;
  std::vector<std::future<void>> futures(num_workers);

  // Start workers
  for (int i = 0; i < num_workers; ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this,
                            &_workers[i], timeseries_a, timeseries_b);
  }

  // wait for workers to be done
  for (auto &future : futures) {
    future.get();
  }

  return SCAMP_NO_ERROR;
}

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads) {
  if (devices.empty() && num_threads == 0) {
    printf("Error: no compute_resources provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  OptionalArgs _opt_args(args->distance_threshold());
  printf("Constructing SCAMP_Operation\n");
  // Construct operation
  SCAMP_Operation op(
      args->timeseries_a().size(), args->timeseries_b().size(), args->window(),
      args->max_tile_size(), devices, !args->has_b(), args->precision_type(),
      args->computing_columns() && args->computing_rows(),
      args->distributed_start_row(), args->distributed_start_col(), _opt_args,
      args->profile_type(), args->mutable_profile_a(),
      args->mutable_profile_b(), args->keep_rows_separate(),
      args->computing_rows(), args->computing_columns(), args->is_aligned(),
      num_threads);
  // Init memory
  op.init();
  start = clock();
  // Execute op
  printf("Starting Join\n");
  if (args->has_b()) {
    op.do_join(args->timeseries_a(), args->timeseries_b());
  } else {
    op.do_join(args->timeseries_a(), args->timeseries_a());
  }
  end = clock();
  printf(
      "Finished %d SCAMP tiles to generate  matrix profile in %f "
      "seconds on %lu devices:\n",
      op.get_completed_tiles(),
      (end - start) / static_cast<double>(CLOCKS_PER_SEC), devices.size());
  op.destroy();
}

}  // namespace SCAMP
