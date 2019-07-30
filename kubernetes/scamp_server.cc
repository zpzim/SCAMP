#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "../src/common.h"
#include "distributed_job.h"
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

constexpr uint64_t TIMEOUT_CHECK_FREQUENCY = 20;

// Job list
std::vector<Job> jobVec;
std::mutex jobVecLock;

void check_time_out() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(TIMEOUT_CHECK_FREQUENCY));
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    for (int i = 0; i < jobVec.size(); i++) {
      if (!jobVec[i].is_done()) {
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
    job.is_done();
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
    if (jobVec[SCAMP_job_id->job_id()].is_done()) {
      *job_result->mutable_args() =
          *jobVec[SCAMP_job_id->job_id()].mutable_args();
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

    MergeProfile(tile->args(), job.mutable_args(), &tile_a, tile->start_col(),
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
  auto f = std::async(std::launch::async, check_time_out);

  RunServer();
  return 0;
}
