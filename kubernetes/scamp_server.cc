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

constexpr uint64_t CLEANUP_CHECK_FREQUENCY = 20;

enum JobListSchedulerType {
  SCHEDULE_TYPE_ISSUE_ORDER = 0,
  SCHEDULE_TYPE_ROUND_ROBIN = 1,
  SCHEDULE_TYPE_LEAST_ETA = 2,
  SCHEDULE_TYPE_LEAST_PROGRESS = 3,
};

class JobList {
 public:
  JobList(JobListSchedulerType t) : schedule_type_(t) {}
  uint64_t add_job(const SCAMPProto::SCAMPArgs &args) {
    std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
    uint64_t job_id = task_list_.size();
    task_list_.emplace(job_id, std::move(Job(args, job_id)));
    run_list_.emplace_back(job_id);
    return job_id;
  }
  Job *get_job(int job_id) {
    std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
    if (task_list_.count(job_id) == 0) {
      return nullptr;
    }
    return &task_list_.at(job_id);
  }
  Job *get_job_to_work_on() {
    std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
    return get_highest_priority_job();
  }
  void cleanup_jobs() {
    std::lock_guard<std::mutex> lockGuard(task_list_mutex_);
    std::vector<int> to_remove;
    auto iter = run_list_.begin();
    while (iter != run_list_.end()) {
      Job &job = task_list_.at(*iter);
      // Check if job timed out
      job.check_time_tile();
      if (job.status() != SCAMPProto::JOB_STATUS_RUNNING) {
        // Remove job from run_list if it has finished
        iter = run_list_.erase(iter);
      } else {
        iter++;
      }
    }
  }

 private:
  Job *get_highest_priority_job() {
    Job *ret = nullptr;
    switch (schedule_type_) {
      case SCHEDULE_TYPE_ISSUE_ORDER:
        for (auto &job_id : run_list_) {
          Job &job = task_list_.at(job_id);
          if (job.has_work()) {
            ret = &job;
            break;
          }
        }
        break;
      case SCHEDULE_TYPE_ROUND_ROBIN:
        for (int i = 0; i < run_list_.size(); ++i) {
          int job_id = run_list_.front();
          run_list_.pop_front();
          run_list_.push_back(job_id);
          Job &job = task_list_.at(job_id);
          if (job.has_work()) {
            ret = &job;
            break;
          }
        }
        break;
      case SCHEDULE_TYPE_LEAST_ETA: {
        bool found = false;
        int best_eta = INT_MAX;
        int best_job;
        for (auto &job_id : run_list_) {
          if (task_list_.at(job_id).has_work()) {
            int eta = task_list_.at(job_id).get_eta();
            if (eta < best_eta) {
              best_eta = eta;
              best_job = job_id;
              found = true;
            }
          }
        }
        if (found) {
          ret = &task_list_.at(best_job);
        }
        break;
      }
      case SCHEDULE_TYPE_LEAST_PROGRESS: {
        bool found = false;
        double best_prog = 0;
        int best_job;
        for (auto &job_id : run_list_) {
          if (task_list_.at(job_id).has_work()) {
            double prog = task_list_.at(job_id).get_progress();
            if (prog > best_prog) {
              best_prog = prog;
              best_job = job_id;
            }
          }
        }
        if (found) {
          ret = &task_list_.at(best_job);
        }
        break;
      }
    }
    return ret;
  }
  std::mutex task_list_mutex_;
  std::unordered_map<int, Job> task_list_;
  std::list<int> run_list_;
  JobListSchedulerType schedule_type_;
};

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
  auto f = std::async(std::launch::async, cleanup_check);

  RunServer();
  return 0;
}
