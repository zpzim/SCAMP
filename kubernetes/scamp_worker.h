#pragma once

#include <grpcpp/grpcpp.h>
#include "scamp.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

class SCAMPWorker {
 public:
  SCAMPWorker(std::shared_ptr<Channel> channel)
      : stub_(SCAMPProto::SCAMPService::NewStub(channel)) {}
  void run();

 private:
  SCAMPProto::SCAMPWork RequestWork(SCAMPProto::SCAMPRequest request);
  SCAMPProto::SCAMPWork ExecuteWork(SCAMPProto::SCAMPWork work);
  SCAMPProto::SCAMPResult MergeResultWithGlobal(
      const SCAMPProto::SCAMPArgs &args);

  SCAMPProto::SCAMPResult ReportFailedTile(
      const SCAMPProto::SCAMPArgs &failed_args);
  std::unique_ptr<SCAMPProto::SCAMPService::Stub> stub_;
};
