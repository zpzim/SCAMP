#pragma once
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
#include "tile.h"
using std::list;
using std::pair;
using std::unordered_map;
using std::vector;

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

namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads);

class SCAMP_Operation {
 private:
  // Precomputed statistics used by tiles
  PrecomputedInfo _precompA, _precompB;

  // Result vectors
  Profile *_profile_a, *_profile_b;

  // Locks for result vectors
  std::mutex _profile_a_lock, _profile_b_lock;

  // Operation specific variables like maximum tile size (see common.h)
  const OpInfo _info;

  // Tile state variables
  // The order to compute the tiles in, set by get_tiles()
  ThreadSafeQueue _work_queue;

  // Lock for counter updates
  std::mutex _counter_lock;

  // The number of completed tiles
  int _completed_tiles;

  // The total number of tiles
  size_t _total_tiles;

  // Cuda devices to compute with
  std::vector<int> _devices;

  // CPU threads to compute with
  int _cpu_workers;

  SCAMPError_t do_tile(SCAMPTileType t, Tile *tile,
                       const google::protobuf::RepeatedField<double> &Ta_h,
                       const google::protobuf::RepeatedField<double> &Tb_h);

  void get_tiles();
  void do_work(const google::protobuf::RepeatedField<double> &timeseries_a,
               const google::protobuf::RepeatedField<double> &timeseries_b,
               const OpInfo *info, const SCAMPArchitecture arch,
               const int device_id);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, bool do_full_join, int64_t start_row,
                  int64_t start_col, OptionalArgs args_,
                  SCAMPProfileType profile_type, Profile *pA, Profile *pB,
                  bool keep_rows, bool compute_rows, bool compute_cols,
                  bool is_aligned, int num_threads)
      : _info(Asize, Bsize, window_sz, max_tile_size, selfjoin, t, start_row,
              start_col, args_, profile_type, keep_rows, compute_rows,
              compute_cols, is_aligned, dev.size() + num_threads),
        _completed_tiles(0),
        _profile_a(pA),
        _profile_b(pB),
        _devices(dev),
        _cpu_workers(num_threads) {}
  SCAMPError_t do_join(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  SCAMPError_t init();
  SCAMPError_t destroy();
  int get_completed_tiles() { return _completed_tiles; }
};

}  // namespace SCAMP
