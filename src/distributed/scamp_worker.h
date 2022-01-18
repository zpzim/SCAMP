#pragma once

#include <grpcpp/grpcpp.h>
#include <queue>
#include "scamp.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

class SCAMPWorker {
 public:
  SCAMPWorker(std::shared_ptr<Channel> channel)
      : stub_(SCAMPProto::SCAMPService::NewStub(channel)) {}
  bool run();

 private:
  double get_expected_throughput();
  void MergeCompletedResults();
  SCAMPProto::SCAMPWork RequestWork(const SCAMPProto::SCAMPRequest &request);
  SCAMPProto::SCAMPWork ExecuteWork(SCAMPProto::SCAMPWork work);
  grpc::Status MergeResultWithGlobal(
      const SCAMPProto::SCAMPArgs &computed_result);

  SCAMPProto::SCAMPResult ReportFailedTile(
      const SCAMPProto::SCAMPArgs &failed_args);
  std::unique_ptr<SCAMPProto::SCAMPService::Stub> stub_;
  std::mutex merge_m_;
  std::queue<std::pair<int, SCAMPProto::SCAMPWork>> work_to_merge_;
};
