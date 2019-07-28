
#include <grpcpp/grpcpp.h>
#include <scamp.grpc.pb.h>
#include "../src/common.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

int do_SCAMP_distributed(SCAMP::SCAMPArgs *args, std::string hostname_port,
                         int64_t distributed_tile_size);

class SCAMPInterface {
 public:
  SCAMPInterface(std::shared_ptr<Channel> channel)
      : stub_(SCAMPProto::SCAMPService::NewStub(channel)) {}
  grpc::Status do_SCAMP_distributed(SCAMPProto::SCAMPArgs *args,
                                    bool async = false);
  void IssueJobsAsync(std::vector<SCAMPProto::SCAMPArgs *> args,
                      int max_wait_seconds);

 private:
  void ManageAsyncJobs();

  static constexpr int max_async_retries_ = 10;
  std::mutex job_m_;
  std::unique_ptr<SCAMPProto::SCAMPService::Stub> stub_;
  std::vector<SCAMPProto::SCAMPStatus> job_statuses_;
};
