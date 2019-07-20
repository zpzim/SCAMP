
#include <grpcpp/grpcpp.h>
#include <scamp.grpc.pb.h>
#include "../src/common.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

int do_SCAMP_distributed(SCAMP::SCAMPArgs *args, std::string hostname_port);

class SCAMPInterface {
 public:
  SCAMPInterface(std::shared_ptr<Channel> channel)
      : stub_(SCAMPProto::SCAMPService::NewStub(channel)) {}
  void do_SCAMP_distributed(SCAMPProto::SCAMPArgs *args);

 private:
  std::unique_ptr<SCAMPProto::SCAMPService::Stub> stub_;
};
