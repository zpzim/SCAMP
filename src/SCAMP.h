#pragma once
#include <cuda.h>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include "SCAMP.pb.h"
#include "SCAMPWorker.h"
#include "common.h"
#include "fft_helper.h"
using std::list;
using std::pair;
using std::unordered_map;
using std::vector;



class ThreadSafeQueue
{
 public:
  
  size_t size() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.size();
  }

  bool empty() {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.empty();

  }
 
  std::pair<int,int> pop()
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    auto item = std::pair<int,int>(-1,-1);
    if (!queue_.empty())
    {
      item = queue_.front();
      queue_.pop();
    }
    return item;
  }
 
 
  void push(const std::pair<int,int>& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
 
  void push(std::pair<int,int>&& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(std::move(item));
    mlock.unlock();
    cond_.notify_one();
  }
 
 private:
  std::queue<std::pair<int,int>> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};    


namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices);

class SCAMP_Operation {
 private:
  PrecomputedInfo _precompA, _precompB;

  // Result vectors
  Profile *_profile_a, *_profile_b;
  
  // Locks for result vectors
  std::mutex _profile_a_lock, _profile_b_lock;  

  // Type of profile to compute
  SCAMPProfileType _profile_type;

  // Total size of A timeseries
  size_t _full_ts_len_A;
  // Total size of B timesereis
  size_t _full_ts_len_B;
  // Max size of the timeseries associated with the tile
  size_t _max_tile_ts_size;
  // Max width of the distance matrix associated with the tile
  size_t _max_tile_width;
  // Max height of the distance matrix associated with the tile
  size_t _max_tile_height;

  // Subsequence window length for MP
  size_t _mp_window;

  // Optional kernel arguments
  OptionalArgs _opt_args;

  // Whether or not we are computing a self join (symmetric distance matrix)
  const bool _self_join;
  // Whether or not to compute MP along the rows.
  const bool _computing_rows;
  // Whether or not to compute MP along the columns.
  const bool _computing_cols;
  // Whether or not time series A and B start with the same prefix.
  const bool _is_aligned;
  // Determines if we should keep the row/column matrix profiles separate or to
  // merge them.
  const bool _keep_rows_separate;

  // Absolute maximum length of a time series to use in a tile
  // TODO(zpzim): Make this the maximum length of the profile in a tile rather
  // than the time series.
  const size_t MAX_TILE_SIZE;

  // Precision type of computation
  const SCAMPPrecisionType _fp_type;

  // CUDA device ids to use for computation
  // TODO(zpzim): Convert this to a general device type (For CPU and GPU
  // computation)
  vector<Worker> _workers;

  // For distributed joins, the start position of this join in relation to other
  // distributed tiles.
  const int64_t _tile_start_row_position;
  const int64_t _tile_start_col_position;

  // Tile state variables
  // The order to compute the tiles in, set by get_tiles()
  ThreadSafeQueue _work_queue;

  // The number of completed tiles
  int _completed_tiles;

  // The total number of tiles
  size_t _total_tiles;

  SCAMPError_t do_tile(SCAMPTileType t, Worker *worker,
                       const google::protobuf::RepeatedField<double> &Ta_h,
                       const google::protobuf::RepeatedField<double> &Tb_h);

  void get_tiles();
  void do_work(Worker *worker, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, bool do_full_join, int64_t start_row,
                  int64_t start_col, OptionalArgs args_,
                  SCAMPProfileType profile_type, Profile *pA, Profile *pB,
                  bool keep_rows, bool compute_rows, bool compute_cols,
                  bool is_aligned)
      : _full_ts_len_A(Asize),
        _mp_window(window_sz),
        MAX_TILE_SIZE(max_tile_size),
        _self_join(selfjoin),
        _completed_tiles(0),
        _fp_type(t),
        _tile_start_row_position(start_row),
        _tile_start_col_position(start_col),
        _opt_args(args_),
        _profile_type(profile_type),
        _profile_a(pA),
        _profile_b(pB),
        _keep_rows_separate(keep_rows),
        _computing_rows(compute_rows),
        _computing_cols(compute_cols),
        _is_aligned(is_aligned) {
    if (_self_join) {
      _full_ts_len_B = _full_ts_len_A;
    } else {
      _full_ts_len_B = Bsize;
    }
    _max_tile_ts_size = std::max(Asize, Bsize) / (dev.size());
    if (_max_tile_ts_size > MAX_TILE_SIZE) {
      _max_tile_ts_size = MAX_TILE_SIZE;
    }
    _max_tile_width = _max_tile_ts_size - _mp_window + 1;
    _max_tile_height = _max_tile_width;
    for (auto device : dev) {
        _workers.emplace_back(_max_tile_width, _max_tile_height, _max_tile_ts_size, _mp_window, _profile_type, device, CUDA_GPU_WORKER, device);
    }
  }
  SCAMPError_t do_join(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  SCAMPError_t init();
  SCAMPError_t destroy();
};

}  // namespace SCAMP
