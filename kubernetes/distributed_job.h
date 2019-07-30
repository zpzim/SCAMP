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

  // Checks tiles to see if any timed out
  void check_time_tile();

  // Returns the total time the job has been running
  int64_t get_elapsed_time();

  // Returns the expected time (in seconds) from now the job will finish
  int64_t get_eta();
  // Returns the ratio of completed tiles to total tiles
  float get_progress();

  // Checks if all the tiles for this job have finished and sets
  // the job status accordingly
  bool is_done();

  // Sets a specific tile to be complete
  void set_tile_finished(int tile_id);

  // Sets a specific tile to the failure state
  void set_tile_failed(int tile_id);

  // Fetches a tile (and data) from the job and returns it in args.
  // Returns false on failure to fetch work
  bool fetch_ready_tile(SCAMPProto::SCAMPArgs *args,
                        const SCAMPProto::SCAMPRequest *request);

  // Gets a specific tile from the job, returns nullptr if there is no such tile
  const DistributedTile *get_tile(int tile_id);

  // Getters
  SCAMPProto::SCAMPArgs *mutable_args() { return &job_args; }
  SCAMPProto::JobStatus status() { return status_; }

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
};
