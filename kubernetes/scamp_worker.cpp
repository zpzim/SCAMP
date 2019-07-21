#include <chrono>
#include <thread>
#include <vector>

#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#endif

#include "../src/SCAMP.h"
#include "../src/common.h"
#include "../src/scamp_exception.h"
#include "../src/scamp_utils.h"
#include "scamp_worker.h"
#include "utils.h"

SCAMPProto::SCAMPWork SCAMPWorker::RequestAndExecuteWork(
    SCAMPProto::SCAMPRequest request) {
  // Container for the data we expect from the server.
  SCAMPProto::SCAMPWork ret;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RequestSCAMPWork(&context, request, &ret);

  // Act upon its status.
  if (status.ok() && ret.valid()) {
    std::cout << "Got work from server" << std::endl;
    SCAMP::SCAMPArgs args;
    auto reply = ret.mutable_args();
    ConvertProtoArgsToSCAMPArgs(*reply, &args);
    std::cout << "Converted arguments to SCAMP format" << std::endl;

    std::cout << "Issuing the following args to SCAMP: " << std::endl;
    args.print();

    InitProfileMemory(&args);
    try {
#ifdef _HAS_CUDA_
      int num_dev;
      vector<int> devices;
      if (cudaGetDeviceCount(&num_dev) == cudaSuccess) {
        for (int i = 0; i < num_dev; ++i) {
          devices.push_back(i);
        }
      }
      if (devices.empty()) {
        do_SCAMP(&args, std::vector<int>(),
                 std::thread::hardware_concurrency());
      } else {
        std::cout << "Starting SCAMP with GPUs" << std::endl;
        do_SCAMP(&args, devices, 0);
      }
#else
      do_SCAMP(&args, std::vector<int>(), std::thread::hardware_concurrency());
#endif
    } catch (SCAMPException e) {
      std::cout << "Error: Problem computing tile: " << e.what() << std::endl;
      ret.set_valid(false);
      return ret;
    }
    auto result = ConvertArgsToReply(args);
    result.set_job_id(reply->job_id());
    result.set_tile_id(reply->tile_id());
    *reply = result;

  } else {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "No work from server" << std::endl;
  }
  return ret;
}

SCAMPProto::SCAMPResult SCAMPWorker::MergeResultWithGlobal(
    const SCAMPProto::SCAMPArgs &args) {
  SCAMPProto::SCAMPResult reply;
  grpc::ClientContext context;

  // The actual RPC.
  Status status = stub_->SCAMPCombiner(&context, args, &reply);

  if (!status.ok()) {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
  }
  return reply;
}

// Worker will act as a slave to the server, requesting work to do ad infinitum
void SCAMPWorker::run() {
  while (true) {
    SCAMPProto::SCAMPRequest r;
    SCAMPProto::SCAMPWork result = RequestAndExecuteWork(r);
    if (result.valid()) {
      SCAMPProto::SCAMPResult res = MergeResultWithGlobal(result.args());
    }
  }
}
