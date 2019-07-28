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

SCAMPProto::SCAMPWork SCAMPWorker::RequestWork(
    SCAMPProto::SCAMPRequest request) {
  // Container for the data we expect from the server.
  SCAMPProto::SCAMPWork ret;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RequestSCAMPWork(&context, request, &ret);

  if (!status.ok()) {
    std::cout << "Could not get work from server: " << status.error_message()
              << std::endl;
  }

  return ret;
}

SCAMPProto::SCAMPWork SCAMPWorker::ExecuteWork(SCAMPProto::SCAMPWork work) {
  SCAMP::SCAMPArgs args;
  auto reply = work.mutable_args();
  ConvertProtoArgsToSCAMPArgs(*reply, &args);
  std::cout << "Converted arguments to SCAMP format" << std::endl;

  std::cout << "Issuing the following args to SCAMP: " << std::endl;
  args.print();

  if (!InitProfileMemory(&args)) {
    std::cout << "Error: Problem allocating memory for matrix profile."
              << std::endl;
    work.set_valid(false);
    return work;
  }
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
      do_SCAMP(&args, std::vector<int>(), std::thread::hardware_concurrency());
    } else {
      std::cout << "Starting SCAMP with GPUs" << std::endl;
      do_SCAMP(&args, devices, 0);
    }
#else
    do_SCAMP(&args, std::vector<int>(), std::thread::hardware_concurrency());
#endif
  } catch (SCAMPException e) {
    std::cout << "Error: Problem computing tile: " << e.what() << std::endl;
    work.set_valid(false);
    return work;
  }

  // Clear out the input data, we don't need it anymore
  args.timeseries_a.clear();
  args.timeseries_b.clear();

  auto result = ConvertArgsToReply(args);
  result.mutable_info()->set_job_id(reply->info().job_id());
  result.mutable_info()->set_tile_id(reply->info().tile_id());
  *reply = result;
  return work;
}

SCAMPProto::SCAMPResult SCAMPWorker::MergeResultWithGlobal(
    const SCAMPProto::SCAMPArgs &computed_result) {
  SCAMPProto::SCAMPResult reply;
  grpc::ClientContext context;

  // The actual RPC.
  Status status = stub_->SCAMPCombiner(&context, computed_result, &reply);

  if (!status.ok()) {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
  }
  return reply;
}

SCAMPProto::SCAMPResult SCAMPWorker::ReportFailedTile(
    const SCAMPProto::SCAMPArgs &failed_args) {
  SCAMPProto::SCAMPResult reply;
  grpc::ClientContext context;

  // The actual RPC.
  Status status = stub_->ReportTileFailure(&context, failed_args, &reply);

  if (!status.ok()) {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
  }
  return reply;
}

float calibration_run(int64_t input_size, std::vector<int> gpus, int threads) {
  SCAMP::SCAMPArgs args = get_default_args(input_size);
  if (!InitProfileMemory(&args)) {
    return -1.0;
  }
  auto begin = std::chrono::high_resolution_clock::now();
  do_SCAMP(&args, gpus, threads);
  auto end = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  return diff.count() / static_cast<double>(1e9);
}

float SCAMPWorker::get_expected_throughput() {
  constexpr int64_t test_input_size_cpu = 1 << 17;
  constexpr int64_t test_input_size_gpu = 1 << 20;
  float time_to_finish;
  int64_t input_size;

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
      input_size = test_input_size_cpu;
      time_to_finish = calibration_run(input_size, std::vector<int>(),
                                       std::thread::hardware_concurrency());
    } else {
      input_size = test_input_size_gpu;
      time_to_finish = calibration_run(input_size, devices, 0);
    }
#else
    input_size = test_input_size_cpu;
    time_to_finish = calibration_run(input_size, std::vector<int>(),
                                     std::thread::hardware_concurrency());
#endif
  } catch (SCAMPException e) {
    std::cout << "Error worker could not execute calibration test: " << e.what()
              << std::endl;
    time_to_finish = -1;
  }
  if (time_to_finish < 0) {
    return -1;
  }
  std::cout << time_to_finish << std::endl;
  return (input_size * input_size / 2) / time_to_finish;
}

// Worker will act as a slave to the server, requesting work to do ad infinitum
bool SCAMPWorker::run() {
  SCAMPProto::SCAMPRequest r;
  float expected_throughput = get_expected_throughput();
  std::cout << expected_throughput << std::endl;
  if (expected_throughput <= 0) {
    return false;
  }
  std::cout << "Worker has throughput of: " << expected_throughput << std::endl;
  r.set_expected_throughput(expected_throughput);
  while (true) {
    SCAMPProto::SCAMPWork work = RequestWork(r);
    // Act upon its status.
    if (work.valid()) {
      // Execute the work using SCAMP
      SCAMPProto::SCAMPWork computed_result = ExecuteWork(work);
      if (computed_result.valid()) {
        // Merge completed result with the server's master copy
        // TODO(zpzim): we can do more work while this is happening
        MergeResultWithGlobal(computed_result.args());
      } else {
        // If execution failed, we need to report this to the server
        ReportFailedTile(work.args());
      }
    } else {
      std::cout << "No work from server" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
}
