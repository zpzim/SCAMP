#include "distributed_job.h"
#include "utils.h"

constexpr uint64_t MAX_TILE_RETRIES = 3;
constexpr uint64_t MIN_TIMEOUT_SECONDS = 10;

void Job::check_time_tile() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &elem : tiles) {
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
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
  return tiles_completed_ / static_cast<float>(tiles.size());
}

// Checks if all the tiles for this job have finished and sets
// the job status accordingly
bool Job::is_done() {
  std::lock_guard<std::mutex> lock(mutex_);
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

// Sets a specific tile to the failure state
void Job::set_tile_failed(int tile_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (tile_id < tiles.size() && tile_id >= 0) {
    tiles[tile_id].set_failed();
  }
}

// Fetches a tile (and data) from the job and returns it in args.
// Returns false on failure to fetch work
bool Job::fetch_ready_tile(SCAMPProto::SCAMPArgs *args,
                           const SCAMPProto::SCAMPRequest *request) {
  std::lock_guard<std::mutex> lock(mutex_);
  // If job is not running do nothing
  if (status_ != SCAMPProto::JOB_STATUS_RUNNING) {
    std::cout << "Fetch Ready tile job not running" << std::endl;
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

bool Job::CombineProfile(const SCAMPProto::SCAMPArgs &request) {
  uint64_t request_width, request_height;
  request_width = request.timeseries_size_a() - request.window() + 1;
  request_height = request.has_b()
                       ? request.timeseries_size_b() - request.window() + 1
                       : request_width;
  uint64_t request_start_row = request.distributed_start_row();
  uint64_t request_start_col = request.distributed_start_col();
  SCAMPProto::Profile tile_a = request.profile_a();
  SCAMPProto::Profile tile_b = request.profile_b();
  int tile_id = request.tile_id();

  std::lock_guard<std::mutex> lock(mutex_);

  if (tiles.count(tile_id) == 0) {
    std::cout << "Combiner trying to use invalid tile." << std::endl;
    return false;
  }
  DistributedTile *tile = &tiles[tile_id];

  if (tile == nullptr) {
    std::cout << "Combiner trying to use invlalid tile." << std::endl;
    return false;
  }

  if (!tile->has_args()) {
    std::cout << "Combiner trying to use uninitialzed tile." << std::endl;
    return false;
  }

  if (tile->status() != TILE_STATUS_RUNNING) {
    std::cout << "Combiner trying to use tile not running." << std::endl;
    return false;
  }

  if (tile->height() != request_height) {
    std::cout << "Combiner request and tile height do not match tile: "
              << tile->height() << " request: " << request_height << std::endl;
    return false;
  }

  if (tile->width() != request_width) {
    std::cout << "Combiner request and tile width do not match tile: "
              << tile->width() << " request: " << request_width << std::endl;
    return false;
  }

  if (tile->start_row() != request_start_row) {
    std::cout << "Combiner request and tile start_row do not match tile: "
              << tile->start_row() << " request: " << request_start_row
              << std::endl;
    return false;
  }

  if (tile->start_col() != request_start_col) {
    std::cout << "Combiner request and tile start_col do not match tile: "
              << tile->start_col() << " request: " << request_start_col
              << std::endl;
    return false;
  }

  if (GetProfileSize(tile_a) != tile->width()) {
    std::cout << "Combiner request and tile profile a sizes do not match tile: "
              << tile->width() << " request: " << GetProfileSize(tile_a)
              << std::endl;
    return false;
  }

  if (tile->args().keep_rows_separate() &&
      GetProfileSize(tile_b) != tile->height()) {
    std::cout << "Combiner request and tile profile b sizes do not match tile: "
              << tile->height() << " request: " << GetProfileSize(tile_b)
              << std::endl;
    return false;
  }

  MergeProfile(tile->args(), &this->job_args, &tile_a, tile->start_col(),
               tile->width(), &tile_b, tile->start_row(), tile->height());

  tile->set_finished();
  tiles_completed_++;
  return true;
}

void Job::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
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

// job's mutex must already be held before calling this method
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
      std::cout << "Tile " << elem.first << " Retry #" << elem.second.retries()
                << std::endl;
      ready_queue.push(&elem.second);
    }
  }
  return true;
}

SCAMPProto::JobStatus Job::status() {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

const SCAMPProto::SCAMPArgs &Job::args() {
  std::lock_guard<std::mutex> lock(mutex_);
  return job_args;
}

bool Job::has_work() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (status_ == SCAMPProto::JOB_STATUS_RUNNING) {
    if (ready_queue.empty()) {
      bool failed = cleanup_failed_tiles();
      return !ready_queue.empty() && !failed;
    }
    return true;
  }
  return false;
}
