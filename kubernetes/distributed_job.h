#pragma once
#include <chrono>
#include <queue>
#include <unordered_map>

#include "distributed_tile.h"
#include "scamp.pb.h"

// Class describing a Distributed SCAMP job
class Job {
 public:
  Job(SCAMPProto::SCAMPArgs args, int id)
      : job_id(id),
        job_args(args),
        status_(SCAMPProto::JOB_STATUS_RUNNING),
        tile_counter(0),
        tiles_completed_(0),
        start_time_(std::chrono::steady_clock::now()) {
    Init();
  }

  Job(Job &&other) {
    std::lock_guard<std::mutex> lock(other.mutex_);
    distributed_tile_size_ = other.distributed_tile_size_;
    tiles_completed_ = other.tiles_completed_;
    job_id = other.job_id;
    tile_counter = other.tile_counter;
    tile_rows = other.tile_rows;
    tile_cols = other.tile_cols;
    distance_matrix_width_ = other.distance_matrix_width_;
    distance_matrix_height_ = other.distance_matrix_height_;
    start_time_ = std::move(other.start_time_);
    end_time_ = std::move(other.end_time_);
    tiles = std::move(other.tiles);
    ready_queue = std::move(other.ready_queue);
    status_ = other.status_;
    job_args = std::move(other.job_args);
  }

  // Checks tiles to see if any timed out
  void check_time_tile();

  // Returns the total time the job has been running
  int64_t get_elapsed_time();

  // Returns the expected time (in seconds) from now the job will finish
  int64_t get_eta();

  // Returns the ratio of completed tiles to total tiles
  float get_progress();

  // Sets a specific tile to the failure state
  void set_tile_failed(int tile_id);

  // Checks if all the tiles for this job have finished and sets
  // the job status accordingly
  bool is_done();

  bool has_work();

  // Fetches a tile (and data) from the job and returns it in args.
  // Returns false on failure to fetch work
  bool fetch_ready_tile(SCAMPProto::SCAMPArgs *args,
                        const SCAMPProto::SCAMPRequest *request);

  // Combines the profile from a set of SCAMPArgs into jobArgs
  bool CombineProfile(const SCAMPProto::SCAMPArgs &request);

  // Getters
  SCAMPProto::JobStatus status();
  const SCAMPProto::SCAMPArgs &args();

 private:
  // Initializes the job using job_args, generates the appropriate tiles and
  // puts them on the ready queue. This function should execute in the Job
  // Constructor
  void Init();

  // Cleans up any failed tiles that need to be retried and puts
  // them back on the ready queue. If they have exceeded the retry
  // count, then we put the job into a failure state and return false.
  bool cleanup_failed_tiles();

  uint64_t distributed_tile_size_;
  int tiles_completed_;
  int job_id;
  int tile_counter;
  int tile_rows;
  int tile_cols;
  int64_t distance_matrix_width_;
  int64_t distance_matrix_height_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;
  std::unordered_map<int, DistributedTile> tiles;
  std::queue<DistributedTile *> ready_queue;
  SCAMPProto::JobStatus status_;

  SCAMPProto::SCAMPArgs job_args;

  std::mutex mutex_;
};
