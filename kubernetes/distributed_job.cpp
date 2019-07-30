#include "distributed_job.h"
#include "utils.h"

constexpr uint64_t MAX_TILE_RETRIES = 3;
constexpr uint64_t MIN_TIMEOUT_SECONDS = 10;

inline int64_t get_current_time() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void Job::check_time_tile() {
  for (auto elem : tiles) {
    if (elem.second.status() == TILE_STATUS_RUNNING) {
      if (get_current_time() - elem.second.start_time() >
          elem.second.timeout()) {
        std::cout << "Tile Timeout ID: " << elem.second.tile_id() << std::endl;
        elem.second.end_time(get_current_time());
        elem.second.status(TILE_STATUS_FAILED);
      }
    }
  }
}
int64_t Job::get_elapsed_time() {
  switch (status_) {
    case SCAMPProto::JOB_STATUS_READY:
      return 0;
    case SCAMPProto::JOB_STATUS_RUNNING:
      return std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::steady_clock::now() - start_time_)
          .count();
    case SCAMPProto::JOB_STATUS_FINISHED:
    case SCAMPProto::JOB_STATUS_FAILED:
      return std::chrono::duration_cast<std::chrono::seconds>(end_time_ -
                                                              start_time_)
          .count();
    case SCAMPProto::JOB_STATUS_INVALID:
    default:
      return -1;
  }
  return -1;
}
int64_t Job::get_eta() {
  int64_t elapsed_time = get_elapsed_time();
  if (elapsed_time < 0) {
    return -1;
  }
  float progress = get_progress();
  if (progress == 0) {
    return -1;
  }
  return (elapsed_time / progress) - elapsed_time;
}
// Returns the ratio of completed tiles to total tiles
float Job::get_progress() {
  return tiles_completed_ / static_cast<float>(tiles.size());
}

// Checks if all the tiles for this job have finished and sets
// the job status accordingly
bool Job::is_done() {
  if (status_ == SCAMPProto::JOB_STATUS_FINISHED) {
    return true;
  }
  for (auto elem : tiles) {
    if (elem.second.status() != TILE_STATUS_FINISHED) {
      return false;
    }
  }
  status_ = SCAMPProto::JOB_STATUS_FINISHED;
  end_time_ = std::chrono::steady_clock::now();
  return true;
}

// Sets a specific tile to be complete
void Job::set_tile_finished(int tile_id) {
  if (tiles.count(tile_id) != 0) {
    tiles[tile_id].status(TILE_STATUS_FINISHED);
    tiles[tile_id].end_time(get_current_time());
    tiles_completed_++;
  }
}

// Sets a specific tile to the failure state
void Job::set_tile_failed(int tile_id) {
  if (tile_id < tiles.size() && tile_id >= 0) {
    tiles[tile_id].status(TILE_STATUS_FAILED);
    tiles[tile_id].end_time(get_current_time());
  }
}

// Fetches a tile (and data) from the job and returns it in args.
// Returns false on failure to fetch work
bool Job::fetch_ready_tile(SCAMPProto::SCAMPArgs *args,
                           const SCAMPProto::SCAMPRequest *request) {
  // If job is not running do nothing
  if (status_ != SCAMPProto::JOB_STATUS_RUNNING) {
    return false;
  }
  if (ready_queue.empty()) {
    // See if there are any failed tiles to retry
    bool failed = cleanup_failed_tiles();

    // Check if we exceeded retry count on a tile (job failed) or we have no
    // work
    if (failed || ready_queue.empty()) {
      return false;
    }
  }

  DistributedTile *tile = ready_queue.front();
  ready_queue.pop();

  // Generate arguments for tile execution
  tile->generate_args(job_args, args);

  // Set timeout
  int64_t timeout = distributed_tile_size_ * distributed_tile_size_ /
                    request->expected_throughput();
  if (!args->has_b()) {
    timeout = timeout / 2;
  }
  timeout = std::max<int64_t>(MIN_TIMEOUT_SECONDS, timeout);
  tile->timeout(timeout);

  // Timer Start
  args->set_job_id(job_id);

  tile->start_time(get_current_time());
  tile->status(TILE_STATUS_RUNNING);
  return true;
}

const DistributedTile *Job::get_tile(int tile_id) {
  if (tiles.count(tile_id) == 0) {
    return nullptr;
  }
  return &tiles[tile_id];
}

void Job::Init() {
  if (!validateArgs(job_args)) {
    status_ = SCAMPProto::JOB_STATUS_INVALID;
    return;
  }
  distance_matrix_width_ =
      job_args.timeseries_a().size() - job_args.window() + 1;
  distance_matrix_height_ =
      job_args.has_b() ? job_args.timeseries_b().size() - job_args.window() + 1
                       : distance_matrix_width_;
  if (!ProfileAllocated(job_args.profile_a())) {
    InitProfile(job_args.mutable_profile_a(), job_args.profile_type(),
                distance_matrix_width_);
  }

  // Make sure the profile size aligns with the distance matrix size
  if (GetProfileSize(job_args.profile_a()) != distance_matrix_width_) {
    status_ = SCAMPProto::JOB_STATUS_INVALID;
    return;
  }

  if (job_args.keep_rows_separate()) {
    if (!ProfileAllocated(job_args.profile_b())) {
      InitProfile(job_args.mutable_profile_b(), job_args.profile_type(),
                  distance_matrix_height_);
    }
    // Make sure the profile size aligns with the distance matrix size
    if (GetProfileSize(job_args.profile_b()) != distance_matrix_height_) {
      status_ = SCAMPProto::JOB_STATUS_INVALID;
      return;
    }
  }

  distributed_tile_size_ = job_args.distributed_tile_size();
  tile_cols = ceil(distance_matrix_width_ /
                   static_cast<double>(distributed_tile_size_));
  if (job_args.has_b()) {
    tile_rows = ceil(distance_matrix_height_ /
                     static_cast<double>(distributed_tile_size_));
  } else {
    tile_rows = tile_cols;
  }
  for (int r = 0; r < tile_rows; r++) {
    for (int c = job_args.has_b() ? 0 : r; c < tile_cols; c++) {
      int64_t height =
          std::min(distributed_tile_size_,
                   distance_matrix_height_ - (r * distributed_tile_size_));
      int64_t width =
          std::min(distributed_tile_size_,
                   distance_matrix_width_ - (c * distributed_tile_size_));
      tiles.emplace(tile_counter,
                    DistributedTile(r, c, height, width, tile_counter));
      ready_queue.push(&tiles[tile_counter]);
      tile_counter++;
    }
  }
  status_ = SCAMPProto::JOB_STATUS_RUNNING;
}

bool Job::cleanup_failed_tiles() {
  for (auto &elem : tiles) {
    if (elem.second.status() == TILE_STATUS_FAILED) {
      if (elem.second.retries() > MAX_TILE_RETRIES) {
        // Tile failed and therefore job has failed
        status_ = SCAMPProto::JOB_STATUS_FAILED;
        end_time_ = std::chrono::steady_clock::now();
        return false;
      }
      // Retry the tile
      elem.second.retries(elem.second.retries() + 1);
      elem.second.start_time(get_current_time());
      elem.second.end_time(INT_MAX);
      ready_queue.push(&elem.second);
    }
  }
  return true;
}
