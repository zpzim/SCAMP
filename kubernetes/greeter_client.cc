/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "helloworld.grpc.pb.h"

#include "../src/SCAMP.h"
#include "../src/common.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
// using helloworld::HelloRequest;
// using helloworld::HelloReply;
using helloworld::Greeter;
using helloworld::SCAMPArgs;
using helloworld::SCAMPRequest;
using helloworld::SCAMPResult;

SCAMP::SCAMPPrecisionType ConvertPrecisionType(
    const helloworld::SCAMPPrecisionType &t) {
  switch (t) {
    case helloworld::PRECISION_DOUBLE:
      return SCAMP::PRECISION_DOUBLE;
    case helloworld::PRECISION_MIXED:
      return SCAMP::PRECISION_MIXED;
    case helloworld::PRECISION_SINGLE:
      return SCAMP::PRECISION_SINGLE;
    default:
      return SCAMP::PRECISION_INVALID;
  }
}

SCAMP::SCAMPProfileType ConvertProfileType(
    const helloworld::SCAMPProfileType &t) {
  switch (t) {
    case helloworld::PROFILE_TYPE_1NN_INDEX:
      return SCAMP::PROFILE_TYPE_1NN_INDEX;
    case helloworld::PROFILE_TYPE_1NN:
      return SCAMP::PROFILE_TYPE_1NN;
    case helloworld::PROFILE_TYPE_SUM_THRESH:
      return SCAMP::PROFILE_TYPE_SUM_THRESH;
    default:
      return SCAMP::PROFILE_TYPE_INVALID;
  }
}

helloworld::Profile ConvertProfile(const SCAMP::Profile &p) {
  // std::cout << "size = " << p.data.size() << std::endl;

  helloworld::Profile out;
  out.set_type(helloworld::PROFILE_TYPE_1NN_INDEX);

  // TODO FIX
  if (p.data.empty()) {
    return out;
  }

  switch (p.type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      *out.mutable_data()->Add()->mutable_uint64_value()->mutable_value() = {
          p.data[0].uint64_value.begin(), p.data[0].uint64_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_1NN:
      *out.mutable_data()->Add()->mutable_float_value()->mutable_value() = {
          p.data[0].float_value.begin(), p.data[0].float_value.end()};
      break;
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      *out.mutable_data()->Add()->mutable_double_value()->mutable_value() = {
          p.data[0].double_value.begin(), p.data[0].double_value.end()};
      break;
  }
  return out;
}

int64_t GetProfileSize(const SCAMP::Profile &p) {
  if (p.data.empty()) {
    return 0;
  }
  switch (p.type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      return p.data[0].uint64_value.size();
    case SCAMP::PROFILE_TYPE_1NN:
      return p.data[0].float_value.size();
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      return p.data[0].double_value.size();
  }
  return 0;
}

helloworld::SCAMPArgs ConvertArgsToReply(const SCAMP::SCAMPArgs &args) {
  helloworld::SCAMPArgs reply;
  *reply.mutable_timeseries_a() = {args.timeseries_a.begin(),
                                   args.timeseries_a.end()};
  *reply.mutable_timeseries_b() = {args.timeseries_b.begin(),
                                   args.timeseries_b.end()};
  *reply.mutable_profile_a() = ConvertProfile(args.profile_a);
  *reply.mutable_profile_b() = ConvertProfile(args.profile_b);
  reply.set_has_b(args.has_b);
  reply.set_window(args.window);
  reply.set_max_tile_size(args.max_tile_size);
  reply.set_distributed_start_row(args.distributed_start_row);
  reply.set_distributed_start_col(args.distributed_start_col);
  reply.set_timeseries_size_a(GetProfileSize(args.profile_a));
  reply.set_timeseries_size_b(GetProfileSize(args.profile_b));
  return reply;
}

class GreeterClient {
 public:
  int randnum;
  std::vector<double> Ta_h, Tb_h;

  GreeterClient() {
    randnum = 0;
    srand(time(NULL));
    randnum = rand();
  }

  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  void InitProfileMemory(SCAMP::SCAMPArgs *args) {
    int FLAGS_window = 100;
    bool FLAGS_keep_rows = false;
    switch (args->profile_type) {
      case SCAMP::PROFILE_TYPE_1NN_INDEX: {
        SCAMP::mp_entry e;
        e.floats[0] = std::numeric_limits<float>::lowest();
        e.ints[1] = -1u;
        args->profile_a.data.emplace_back();
        args->profile_a.data[0].uint64_value.resize(
            args->timeseries_a.size() - FLAGS_window + 1, e.ulong);
        if (args->keep_rows_separate) {
          args->profile_b.data.emplace_back();
          args->profile_b.data[0].uint64_value.resize(
              args->timeseries_b.size() - FLAGS_window + 1, e.ulong);
        }
      }
      case SCAMP::PROFILE_TYPE_1NN: {
        args->profile_a.data.emplace_back();
        args->profile_a.data[0].float_value.resize(
            args->timeseries_a.size() - FLAGS_window + 1,
            std::numeric_limits<float>::lowest());
        if (args->keep_rows_separate) {
          args->profile_b.data.emplace_back();
          args->profile_b.data[0].float_value.resize(
              args->timeseries_b.size() - FLAGS_window + 1,
              std::numeric_limits<float>::lowest());
        }
      }
      case SCAMP::PROFILE_TYPE_SUM_THRESH: {
        args->profile_a.data.emplace_back();
        args->profile_a.data[0].double_value.resize(
            args->timeseries_a.size() - FLAGS_window + 1, 0);
        if (args->keep_rows_separate) {
          args->profile_b.data.emplace_back();
          args->profile_b.data[0].double_value.resize(
              args->timeseries_b.size() - FLAGS_window + 1, 0);
        }
      }
      default:
        break;
    }
  }

  helloworld::SCAMPWork RequestSCAMPWork(SCAMPRequest request) {
    // Container for the data we expect from the server.
    helloworld::SCAMPWork ret;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // std::cout << "Client requests work from server" << std::endl;

    // The actual RPC.
    Status status = stub_->RequestSCAMPWork(&context, request, &ret);

    // Act upon its status.
    if (status.ok() && ret.valid()) {
      SCAMPArgs *reply = ret.mutable_args();

      // std::cout << "client got ok back from server" << std::endl;

      for (int i = 0; i < reply->timeseries_size_a(); i++) {
        Ta_h.push_back(reply->timeseries_a()[i]);
      }

      for (int i = 0; i < reply->timeseries_size_b(); i++) {
        Tb_h.push_back(reply->timeseries_b()[i]);
      }

      /*
      std::cout << "ta_h size: " << Ta_h.size() << std::endl;
      std::cout << "tb_h size: " << Tb_h.size() << std::endl;
      std::cout << "argtimeseriesa size: " << reply->timeseries_size_a()
                << std::endl;
      */

      /*
      for(int i = 0; i < reply.timeseries_size_a(); i++)
        {
          std::cout << "ta_h: " << Ta_h[i] << std::endl;
        }
      */

      // = getSCAMPArgsFromProto(reply);
      // SCAMP::SCAMPArgs args = getSCAMPArgsFromProto(reply);

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

      // std::cout << "CLIENT INITPROFILEMEMEORYSTART" << std::endl <<
      // std::flush;
      InitProfileMemory(&args);
      // std::cout << "CLIENT DO_SCAMP START" << std::endl << std::flush;
      do_SCAMP(&args, std::vector<int>(), std::thread::hardware_concurrency());

      auto result = ConvertArgsToReply(args);
      result.set_job_id(reply->job_id());
      result.set_tile_id(reply->tile_id());
      *reply = result;
      // std::cout << "CLIENT DO_SCAMP FINISH" << std::endl << std::flush;

      // std::cout << "width: " << args.max_tile_size << std::endl;

    } else {
      // FIXME Remove
      usleep(2 * 1000000);
      std::cout << "No work from server" << std::endl;
    }
    Ta_h.clear();
    Tb_h.clear();
    return ret;
  }

  SCAMPResult SCAMPCombiner(const SCAMPArgs &args) {
    SCAMPResult reply;
    ClientContext context;

    // std::cout << "client scampcombiner" << std::endl;

    // The actual RPC.
    Status status = stub_->SCAMPCombiner(&context, args, &reply);

    // std::cout << "clietn scampcombiner222222" << std::endl;

    if (status.ok()) {
      // return reply.done();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      // return -1;

      // FIXME remove
      usleep(2 * 1000000);
    }
    return reply;
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char **argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  // std::cout << "client start" << std::endl;

  char *port;
  char *ip;

  port = getenv("SERVERVEC_SERVICE_PORT");
  ip = getenv("SERVERVEC_SERVICE_HOST");

  if (ip != NULL) {
    std::cout << "ip: " << ip << std::endl;
    std::cout << "port: " << port << std::endl;
  }

  // run local
  // ip = "localhost";
  // port = "30078";
  std::string newip = "localhost";
  std::string newport = "30078";

  // std::string good = std::string(ip) + ":" + std::string(port);
  std::string good = newip + ":" + newport;

  GreeterClient greeter(
      grpc::CreateChannel(good, grpc::InsecureChannelCredentials()));

  while (true) {
    SCAMPRequest r;
    helloworld::SCAMPWork work = greeter.RequestSCAMPWork(r);
    if (work.valid()) {
      SCAMPResult res = greeter.SCAMPCombiner(work.args());
    }
  }

  return 0;
}
