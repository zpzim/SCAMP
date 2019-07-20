#include <chrono>
#include <thread>
#include <vector>

#include "../src/SCAMP.h"
#include "../src/common.h"
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
    SCAMPProto::SCAMPArgs *reply = ret.mutable_args();
    std::vector<double> Ta_h, Tb_h;

    for (int i = 0; i < reply->timeseries_size_a(); i++) {
      Ta_h.push_back(reply->timeseries_a()[i]);
    }

    for (int i = 0; i < reply->timeseries_size_b(); i++) {
      Tb_h.push_back(reply->timeseries_b()[i]);
    }

    SCAMP::SCAMPArgs args;
    args.max_tile_size = reply->max_tile_size();
    args.distributed_start_row = reply->distributed_start_row();
    args.distributed_start_col = reply->distributed_start_col();
    args.distance_threshold = reply->distance_threshold();
    args.computing_columns = reply->computing_columns();
    args.computing_rows = reply->computing_rows();
    args.profile_a.type = ConvertProfileType(reply->profile_type());
    args.profile_b.type = ConvertProfileType(reply->profile_type());
    args.profile_type = ConvertProfileType(reply->profile_type());
    args.precision_type = ConvertPrecisionType(reply->precision_type());
    args.keep_rows_separate = reply->keep_rows_separate();
    args.is_aligned = reply->is_aligned();
    args.window = reply->window();
    args.has_b = reply->has_b();
    args.timeseries_a = std::move(Ta_h);
    args.timeseries_b = std::move(Tb_h);

    InitProfileMemory(&args);

    do_SCAMP(&args, std::vector<int>(), std::thread::hardware_concurrency());

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

// Worker will act as a client to the SCAMPserver/Workers, request a job and
// wait for its completion
void SCAMPWorker::do_SCAMP_distributed(SCAMPProto::SCAMPArgs *args) {
  grpc::ClientContext context;
  SCAMPProto::SCAMPStatus status;
  stub_->IssueNewJob(&context, *args, &status);
  SCAMPProto::SCAMPJobID id;
  id.set_job_id(status.job_id());
  bool done = false;
  while (!done) {
    stub_->CheckJobStatus(&context, id, &status);
    if (status.status() == SCAMPProto::JOB_STATUS_FINISHED) {
      done = true;
    }
  }
  SCAMPProto::SCAMPWork result;
  stub_->FetchJobResult(&context, id, &result);
  *args = result.args();
}
