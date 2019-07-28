#include <chrono>
#include <thread>
#include <vector>

#include "../src/SCAMP.h"
#include "../src/common.h"
#include "scamp_interface.h"
#include "utils.h"

void SCAMPInterface::IssueJobsAsync(std::vector<SCAMPProto::SCAMPArgs *> args,
                                    int max_wait_seconds) {
  while (true) {
    auto iter = args.begin();
    while (iter != args.end()) {
      int retries = 0;
      grpc::Status s = this->do_SCAMP_distributed(*iter, true);
      if (!s.ok()) {
        std::cout << "Problem issuing async job: " << s.error_message()
                  << std::endl;
        while (s.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
          if (retries > max_async_retries_) {
            break;
          }
          retries += 1;
          std::cout << "Issue job failed. Attempt #" << retries
                    << ", will wait " << 60 * retries
                    << " seconds before attempting again." << std::endl;
          std::this_thread::sleep_for(std::chrono::seconds(60 * retries));
          s = this->do_SCAMP_distributed(*iter, true);
        }
        if (retries > max_async_retries_) {
          std::cout << "Could not issue job: " << (**iter).DebugString()
                    << std::endl;
        }
      }
    }
  }
}

void SCAMPInterface::ManageAsyncJobs() {
  while (true) {
    SCAMPProto::SCAMPJobID id;
    SCAMPProto::SCAMPStatus status;
    grpc::Status s;
    {
      std::unique_lock<std::mutex> lock(job_m_);
      for (auto &elem : job_statuses_) {
        if (elem.status() != SCAMPProto::JOB_STATUS_RUNNING) {
          continue;
        }
        id.set_job_id(elem.job_id());
        {
          grpc::ClientContext context;
          s = stub_->CheckJobStatus(&context, id, &status);
        }
        if (!s.ok()) {
          std::cout << "Error updating job status for job " << elem.job_id()
                    << ": " << s.error_message() << std::endl;
          continue;
        }
        elem = status;
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }
}

// Worker will act as a client to the SCAMPserverWorkers, request a job and wait
// for its completion if async is not specified
grpc::Status SCAMPInterface::do_SCAMP_distributed(SCAMPProto::SCAMPArgs *args,
                                                  bool async) {
  SCAMPProto::SCAMPStatus status;
  grpc::Status s;
  {
    grpc::ClientContext context;
    s = stub_->IssueNewJob(&context, *args, &status);
  }
  if (!s.ok()) {
    std::cout << "Error issuing SCAMP Job: " << s.error_message() << std::endl;
    return s;
  }
  std::cout << "Issued Job: " << status.job_id() << std::endl;
  if (async) {
    std::unique_lock<std::mutex> lock(job_m_);
    job_statuses_.push_back(std::move(status));
    return s;
  }
  SCAMPProto::SCAMPJobID id;
  id.set_job_id(status.job_id());
  // id.set_job_id(0);
  bool done = false;
  while (!done) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    {
      grpc::ClientContext context;
      s = stub_->CheckJobStatus(&context, id, &status);
    }
    if (!s.ok()) {
      std::cout << "Error Checking Job Status. " << s.error_message()
                << std::endl;
      continue;
    }
    std::cout << status.DebugString() << std::endl;
    if (status.status() == SCAMPProto::JOB_STATUS_FAILED) {
      // TODO(zpzim): Deal with failure
    }
    if (status.status() == SCAMPProto::JOB_STATUS_FINISHED) {
      done = true;
    }
  }
  SCAMPProto::SCAMPWork result;
  {
    grpc::ClientContext context;
    s = stub_->FetchJobResult(&context, id, &result);
  }
  if (!s.ok()) {
    std::cout << "Error fetching job Result: " << s.error_message()
              << std::endl;
    return s;
  } else {
    std::cout << "Fetched Job Result" << std::endl;
  }
  *args = result.args();
  return s;
}

int do_SCAMP_distributed(SCAMP::SCAMPArgs *args, std::string hostname_port,
                         int64_t distributed_tile_size) {
  SCAMPProto::SCAMPArgs proto_args = ConvertArgsToReply(*args);

  proto_args.mutable_info()->set_distributed_tile_size(distributed_tile_size);

  grpc::ChannelArguments ch_args;

  // Do not limit input size
  ch_args.SetMaxReceiveMessageSize(-1);

  SCAMPInterface worker(grpc::CreateCustomChannel(
      hostname_port, grpc::InsecureChannelCredentials(), ch_args));

  worker.do_SCAMP_distributed(&proto_args, false);

  ConvertProtoArgsToSCAMPArgs(proto_args, args);
}
