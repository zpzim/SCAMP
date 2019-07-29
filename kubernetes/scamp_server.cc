#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "../src/common.h"
#include "distributed_tile.h"
#include "scamp.grpc.pb.h"
#include "utils.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using SCAMPProto::Profile;
using SCAMPProto::SCAMPArgs;
using SCAMPProto::SCAMPRequest;
using SCAMPProto::SCAMPResult;
using SCAMPProto::SCAMPStatus;

using SCAMPProto::SCAMPService;

class Job;

constexpr uint64_t MAX_TILE_RETRIES = 3;
constexpr uint64_t TIMEOUT_CHECK_FREQUENCY = 20;
constexpr uint64_t MIN_TIMEOUT_SECONDS = 10;

// Job list
std::vector<Job> jobVec;
std::mutex jobVecLock;

// Class describing a Distributed SCAMP job
class Job {
 public:
  Job(SCAMPArgs args, int id)
      : job_id(id),
        job_args(args),
        status_(SCAMPProto::JOB_STATUS_RUNNING),
        tile_counter(0),
        tiles_completed_(0),
        start_time_(std::chrono::steady_clock::now()) {
    Init();
  }

  void check_time_tile() {
    for (auto elem : tiles) {
      if (elem.second.status() == TILE_STATUS_RUNNING) {
        if ((time(0) - elem.second.start_time()) > elem.second.timeout()) {
          std::cout << "Tile Timeout ID: " << elem.second.tile_id()
                    << std::endl;
          elem.second.end_time(time(0));
          elem.second.status(TILE_STATUS_FAILED);
        }
      }
    }
  }
  int64_t get_elapsed_time() {
    switch (status_) {
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
        return -1;
    }
  }

  int64_t get_eta() {
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
  float get_progress() {
    return tiles_completed_ / static_cast<float>(tiles.size());
  }

  // Checks if all the tiles for this job have finished and sets
  // the job status accordingly
  bool job_done() {
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
  void set_tile_finished(int tile_id) {
    if (tiles.count(tile_id)) {
      tiles[tile_id].status(TILE_STATUS_FINISHED);
      tiles[tile_id].end_time(time(0));
      tiles_completed_++;
    }
  }

  // Sets a specific tile to the failure state
  void set_tile_failed(int tile_id) {
    if (tile_id < tiles.size() && tile_id >= 0) {
      tiles[tile_id].status(TILE_STATUS_FAILED);
      tiles[tile_id].end_time(time(0));
    }
  }

  // Fetches a tile (and data) from the job and returns it in args.
  // Returns false on failure to fetch work
  bool fetch_ready_tile(SCAMPArgs *args, const SCAMPRequest *request) {
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
    tile->start_time(time(0));
    tile->status(TILE_STATUS_RUNNING);
    return true;
  }

  SCAMPArgs *get_job_args() { return &job_args; }
  const DistributedTile *get_tile(int tile_id) {
    if (!tiles.count(tile_id)) {
      return nullptr;
    }
    return &tiles[tile_id];
  }
  SCAMPProto::JobStatus status() { return status_; }

 private:
  void Init() {
    if (!validateArgs(job_args)) {
      status_ = SCAMPProto::JOB_STATUS_INVALID;
      return;
    }
    distance_matrix_width_ =
        job_args.timeseries_a().size() - job_args.window() + 1;
    distance_matrix_height_ =
        job_args.has_b()
            ? job_args.timeseries_b().size() - job_args.window() + 1
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
      for (int c = 0; c < tile_cols; c++) {
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

  // Cleans up any failed tiles that need to be retried and puts
  // them back on the ready queue. If they have exceeded the retry
  // count, then we put the job into a failure state and return false.
  bool cleanup_failed_tiles() {
    for (auto &elem : tiles) {
      if (elem.second.status() == TILE_STATUS_FAILED) {
        if (elem.second.retries() > MAX_TILE_RETRIES) {
          status_ = SCAMPProto::JOB_STATUS_FAILED;
          end_time_ = std::chrono::steady_clock::now();
          return false;
        } else {
          elem.second.retries(elem.second.retries() + 1);
          elem.second.start_time(time(0));
          elem.second.end_time(INT_MAX);
          ready_queue.push(&elem.second);
        }
      }
    }
    return true;
  }

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

  // This is better as a set (key is tile id)
  SCAMPArgs job_args;
};

void check_time_out() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(TIMEOUT_CHECK_FREQUENCY));
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    for (int i = 0; i < jobVec.size(); i++) {
      if (!jobVec[i].job_done()) {
        jobVec[i].check_time_tile();
      }
    }
  }
}

// Logic and data behind the server's behavior.
class SCAMPServiceImpl final : public SCAMPService::Service {
 public:
 private:
  // Takes a SCAMPArgs proto and tries to create a SCAMP job and add it to the
  // jobVec
  // Returns a job id in SCAMPStatus if we create a job
  // Returns failure state in SCAMPStatus if we fail to create a job
  Status IssueNewJob(ServerContext *context,
                     const SCAMPProto::SCAMPArgs *job_args,
                     SCAMPStatus *status) override {
    if (!validateArgs(*job_args)) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Invalid SCAMP arguments");
    }
    try {
      std::lock_guard<std::mutex> lockGuard(jobVecLock);
      uint64_t cur_job_id = jobVec.size();
      jobVec.emplace_back(*job_args, cur_job_id);
      status->set_status(jobVec.back().status());
      status->set_job_id(cur_job_id);
    } catch (const std::bad_alloc &e) {
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, e.what());
    } catch (const std::exception &e) {
      return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
    return Status::OK;
  }

  // Takes a job_id and returns the status of the job associated with that ID
  Status CheckJobStatus(ServerContext *context,
                        const SCAMPProto::SCAMPJobID *SCAMP_job_id,
                        SCAMPStatus *status) override {
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    status->set_job_id(SCAMP_job_id->job_id());
    std::cout << "Checking status of job id: " << SCAMP_job_id->job_id()
              << " jobvec size = " << jobVec.size() << std::endl;
    if (SCAMP_job_id->job_id() >= jobVec.size() || SCAMP_job_id->job_id() < 0) {
      status->set_status(SCAMPProto::JOB_STATUS_INVALID);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    Job &job = jobVec[SCAMP_job_id->job_id()];
    job.job_done();
    status->set_status(job.status());
    status->set_progress(job.get_progress());
    status->set_time_elapsed(job.get_elapsed_time());
    status->set_eta(job.get_eta());
    return Status::OK;
  }

  // Takes a job id and returns the completed work associated with that id if
  // the job has been completed
  // Otherwise returns a null result
  Status FetchJobResult(ServerContext *context,
                        const SCAMPProto::SCAMPJobID *SCAMP_job_id,
                        SCAMPProto::SCAMPWork *job_result) override {
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    if (SCAMP_job_id->job_id() >= jobVec.size() || SCAMP_job_id->job_id() < 0) {
      job_result->set_valid(false);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    if (jobVec[SCAMP_job_id->job_id()].job_done()) {
      *job_result->mutable_args() =
          *jobVec[SCAMP_job_id->job_id()].get_job_args();
      job_result->set_valid(true);
    } else {
      job_result->set_valid(false);
    }
    return Status::OK;
  }

  // RPC called by workers. Fetches the next tile and sends it to the worker.
  Status RequestSCAMPWork(ServerContext *context, const SCAMPRequest *request,
                          SCAMPProto::SCAMPWork *reply) override {
    std::cout << "Work requested from server" << std::endl;
    SCAMPArgs *args = reply->mutable_args();
    std::lock_guard<std::mutex> lockGuard(jobVecLock);

    for (int i = 0; i < jobVec.size(); i++) {
      if (jobVec[i].fetch_ready_tile(args, request)) {
        std::cout << "Tile fetched from job " << i << std::endl;
        reply->set_valid(true);
        return Status::OK;
      }
    }
    reply->set_valid(false);
    return Status::OK;
  }

  // RPC called by workers. Retrieves completed work from worker and merges it
  // with the global profile associated with that job
  Status SCAMPCombiner(ServerContext *context, const SCAMPArgs *request,
                       SCAMPResult *reply) override {
    uint64_t request_width, request_height;
    request_width = request->timeseries_size_a() - request->window() + 1;
    request_height = request->has_b()
                         ? request->timeseries_size_b() - request->window() + 1
                         : request_width;
    uint64_t request_start_row = request->distributed_start_row();
    uint64_t request_start_col = request->distributed_start_col();
    SCAMPProto::Profile tile_a = request->profile_a();
    SCAMPProto::Profile tile_b = request->profile_b();
    int job_id = request->job_id();
    int tile_id = request->tile_id();

    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    if (job_id >= jobVec.size() || job_id < 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    Job &job = jobVec[job_id];

    const DistributedTile *tile = job.get_tile(tile_id);
    if (tile == nullptr) {
      std::cout << "Combiner trying to use invlalid tile." << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Invalid Tile ID");
    }

    if (!tile->has_args()) {
      std::cout << "Combiner trying to use uninitialzed tile." << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Invalid Tile ID");
    }

    if (tile->status() != TILE_STATUS_RUNNING) {
      std::cout << "Combiner trying to use tile not running." << std::endl;
      return grpc::Status(grpc::StatusCode::ABORTED,
                          "Execution was cancelled by server!");
    }

    if (tile->height() != request_height) {
      std::cout << "Combiner request and tile height do not match tile: "
                << tile->height() << " request: " << request_height
                << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter height mismatch");
    }

    if (tile->width() != request_width) {
      std::cout << "Combiner request and tile width do not match tile: "
                << tile->width() << " request: " << request_width << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter width mismatch");
    }

    if (tile->start_row() != request_start_row) {
      std::cout << "Combiner request and tile start_row do not match tile: "
                << tile->start_row() << " request: " << request_start_row
                << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter start row mismatch");
    }

    if (tile->start_col() != request_start_col) {
      std::cout << "Combiner request and tile start_col do not match tile: "
                << tile->start_col() << " request: " << request_start_col
                << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter start_col mismatch");
    }

    if (GetProfileSize(tile_a) != tile->width()) {
      std::cout
          << "Combiner request and tile profile a sizes do not match tile: "
          << tile->width() << " request: " << GetProfileSize(tile_a)
          << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter profile a size mismatch");
    }

    if (tile->args().keep_rows_separate() &&
        GetProfileSize(tile_b) != tile->height()) {
      std::cout
          << "Combiner request and tile profile b sizes do not match tile: "
          << tile->height() << " request: " << GetProfileSize(tile_b)
          << std::endl;
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Tile parameter profile b size  mismatch");
    }

    MergeProfile(tile->args(), job.get_job_args(), &tile_a, tile->start_col(),
                 tile->width(), &tile_b, tile->start_row(), tile->height());

    job.set_tile_finished(tile_id);

    std::cout << "Finished Merging" << std::endl;
    return Status::OK;
  }

  // RPC called by workers when tile execution fails
  Status ReportTileFailure(ServerContext *context, const SCAMPArgs *request,
                           SCAMPResult *reply) override {
    int job_id = request->job_id();
    int tile_id = request->tile_id();
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    if (job_id >= jobVec.size() || job_id < 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    jobVec[job_id].set_tile_failed(tile_id);
    return Status::OK;
  }
};

void RunServer() {
  std::string server_address("localhost:30078");

  SCAMPServiceImpl service;

  ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  std::cout << "Add Listening Port" << std::endl;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Do not limit input size
  builder.SetMaxReceiveMessageSize(INT_MAX);

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  std::cout << "Register Service" << std::endl;
  builder.RegisterService(&service);

  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  if (server == nullptr) {
    std::cout << "Error building server." << std::endl;
    exit(1);
  }
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char **argv) {
  std::thread check_time_out();

  RunServer();
  return 0;
}
