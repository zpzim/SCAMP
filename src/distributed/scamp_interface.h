#pragma once

#include <grpcpp/grpcpp.h>
#include <scamp.grpc.pb.h>

#include "common/scamp_args.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

void do_SCAMP_distributed(SCAMP::SCAMPArgs *args,
                          const std::string &hostname_port,
                          int64_t distributed_tile_size);

class SCAMPInterface {
 public:
  SCAMPInterface(std::shared_ptr<Channel> channel)
      : stub_(SCAMPProto::SCAMPService::NewStub(channel)) {}
  void do_SCAMP_distributed(SCAMPProto::SCAMPArgs *args);

 private:
  std::unique_ptr<SCAMPProto::SCAMPService::Stub> stub_;
};
