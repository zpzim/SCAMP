#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "distributed_job.h"
#include "job_list.h"
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

constexpr uint64_t CLEANUP_CHECK_FREQUENCY = 20;

static JobList jobs(SCHEDULE_TYPE_ROUND_ROBIN);

void cleanup_check() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(CLEANUP_CHECK_FREQUENCY));
    jobs.cleanup_jobs();
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
      uint64_t curr_job_id = jobs.add_job(*job_args);
      Job *job = jobs.get_job(curr_job_id);
      status->set_status(job->status());
      status->set_job_id(curr_job_id);
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
    status->set_job_id(SCAMP_job_id->job_id());
    Job *job = jobs.get_job(SCAMP_job_id->job_id());
    if (job == nullptr) {
      status->set_status(SCAMPProto::JOB_STATUS_INVALID);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    job->is_done();
    status->set_status(job->status());
    status->set_progress(job->get_progress());
    status->set_time_elapsed(job->get_elapsed_time());
    status->set_eta(job->get_eta());
    return Status::OK;
  }

  // Takes a job id and returns the completed work associated with that id if
  // the job has been completed
  // Otherwise returns a null result
  Status FetchJobResult(ServerContext *context,
                        const SCAMPProto::SCAMPJobID *SCAMP_job_id,
                        SCAMPProto::SCAMPWork *job_result) override {
    Job *job = jobs.get_job(SCAMP_job_id->job_id());
    if (job == nullptr) {
      job_result->set_valid(false);
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    if (job->is_done()) {
      *job_result->mutable_args() = job->args();
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
    Job *job = jobs.get_job_to_work_on();
    if (job == nullptr) {
      reply->set_valid(false);
      return Status::OK;
    }
    if (!job->fetch_ready_tile(args, request)) {
      reply->set_valid(false);
      return Status::OK;
    }
    reply->set_valid(true);
    return Status::OK;
  }

  // RPC called by workers. Retrieves completed work from worker and merges it
  // with the global profile associated with that job
  Status SCAMPCombiner(ServerContext *context, const SCAMPArgs *request,
                       SCAMPResult *reply) override {
    int job_id = request->job_id();
    Job *job = jobs.get_job(job_id);
    if (job == nullptr) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }

    std::cout << "Start Merge" << std::endl;
    job->CombineProfile(*request);
    std::cout << "Finished Merging" << std::endl;
    return Status::OK;
  }

  // RPC called by workers when tile execution fails
  Status ReportTileFailure(ServerContext *context, const SCAMPArgs *request,
                           SCAMPResult *reply) override {
    int job_id = request->job_id();
    int tile_id = request->tile_id();
    Job *job = jobs.get_job(job_id);
    if (job == nullptr) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid Job ID");
    }
    job->set_tile_failed(tile_id);
    return Status::OK;
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:30078");

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
  auto f = std::async(std::launch::async, cleanup_check);

  RunServer();
  return 0;
}
