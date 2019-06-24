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

/*
message SCAMPArgs {
    repeated double timeseries_a = 1;
    repeated double timeseries_b = 2;
    Profile profile_a = 3;
    Profile profile_b = 4;
    bool has_b = 5;
    uint64 window = 6;
    uint64 max_tile_size = 7;
    int64 distributed_start_row = 8;
    int64 distributed_start_col = 9;
    double distance_threshold = 10;
    SCAMPPrecisionType precision_type = 11;
    SCAMPProfileType profile_type = 12;
    bool computing_rows = 13;
    bool computing_columns = 14;
    bool keep_rows_separate = 15;
    bool is_aligned = 16;
    int64 timeseries_size_a = 17;
    int64 timeseries_size_b = 18;
    int64 job_id = 19;
}

message RepeatedDouble {
    repeated double value = 1;
}

message ProfileData {
    oneof Data {
        RepeatedUInt uint32_value = 1;
        RepeatedULong uint64_value = 2;
        RepeatedFloat float_value = 3;
        RepeatedDouble double_value = 4;
    }
}

message Profile {
    repeated ProfileData data = 1;
    SCAMPProfileType type = 2;
}

struct ProfileData {
  // Only one of these should be set at once
  std::vector<uint32_t> uint32_value;
  std::vector<uint64_t> uint64_value;
  std::vector<float> float_value;
  std::vector<double> double_value;
};

// Stores information about a matrix profile
struct Profile {
  std::vector<ProfileData> data;
  SCAMPProfileType type;
};




// Arguments for a SCAMP operation
// This is an external user's interface to the SCAMP library
struct SCAMPArgs {
  std::vector<double> timeseries_a;
  std::vector<double> timeseries_b;
  Profile profile_a;
  Profile profile_b;
  bool has_b;
  uint64_t window;
  uint64_t max_tile_size;
  int64_t distributed_start_row;
  int64_t distributed_start_col;
  double distance_threshold;
  SCAMPPrecisionType precision_type;
  SCAMPProfileType profile_type;
  bool computing_rows;
  bool computing_columns;
  bool keep_rows_separate;
  bool is_aligned;
};

*/

helloworld::Profile ConvertProfile(const SCAMP::Profile &p) {
  // printf("size = %d\n", p.data.size());

  std::cout << "size = " << p.data.size() << std::endl;

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
  printf("Hello1\n");
  // reply.set_timeseries_b(args.timeseries_b);

  *reply.mutable_profile_a() = ConvertProfile(args.profile_a);
  *reply.mutable_profile_b() = ConvertProfile(args.profile_b);
  printf("Hello2\n");
  reply.set_has_b(args.has_b);
  reply.set_window(args.window);
  reply.set_max_tile_size(args.max_tile_size);
  reply.set_distributed_start_row(args.distributed_start_row);
  reply.set_distributed_start_col(args.distributed_start_col);
  printf("Hello3\n");
  reply.set_timeseries_size_a(GetProfileSize(args.profile_a));
  reply.set_timeseries_size_b(GetProfileSize(args.profile_b));
  printf("Hello4\n");
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
        if (FLAGS_keep_rows) {
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
        if (FLAGS_keep_rows) {
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
        if (FLAGS_keep_rows) {
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

    std::cout << "Client requests work from server" << std::endl;

    // The actual RPC.
    Status status = stub_->RequestSCAMPWork(&context, request, &ret);

    // Act upon its status.
    if (status.ok() && ret.valid()) {
      SCAMPArgs *reply = ret.mutable_args();

      std::cout << "client got ok back from server" << std::endl;

      for (int i = 0; i < reply->timeseries_size_a(); i++) {
        Ta_h.push_back(reply->timeseries_a()[i]);
      }

      for (int i = 0; i < reply->timeseries_size_b(); i++) {
        Tb_h.push_back(reply->timeseries_b()[i]);
      }

      std::cout << "ta_h size: " << Ta_h.size() << std::endl;
      std::cout << "tb_h size: " << Tb_h.size() << std::endl;
      std::cout << "argtimeseriesa size: " << reply->timeseries_size_a()
                << std::endl;

      /*
      for(int i = 0; i < reply.timeseries_size_a(); i++)
        {
          std::cout << "ta_h: " << Ta_h[i] << std::endl;
        }
      */

      // = getSCAMPArgsFromProto(reply);
      // SCAMP::SCAMPArgs args = getSCAMPArgsFromProto(reply);

      SCAMP::SCAMPArgs args;
      // args.job_id = reply.job_id();
      // args.timeseries_size_a = reply.timeseries_size_a();
      // args.timeseries_size_b = reply.timeseries_size_b();
      args.max_tile_size = reply->max_tile_size();
      args.distributed_start_row = reply->distributed_start_row();
      args.distributed_start_col = reply->distributed_start_col();
      args.distance_threshold = reply->distance_threshold();
      args.computing_columns = reply->computing_columns();
      args.computing_rows = reply->computing_rows();
      /*
      args.profile_a.type = reply.profile_a();
      args.profile_b.type = reply.profile_b();
      args.precision_type = reply.precision_type();
      args.profile_type = reply.profile_type();
      */
      // TODO: fix
      args.profile_a.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
      args.profile_b.type = SCAMP::PROFILE_TYPE_1NN_INDEX;
      args.precision_type = SCAMP::PRECISION_DOUBLE;
      args.profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;

      args.keep_rows_separate = reply->keep_rows_separate();
      args.is_aligned = reply->is_aligned();
      args.window = reply->window();
      args.has_b = reply->has_b();
      args.timeseries_a = std::move(Ta_h);

      std::cout << "CLIENT INITPROFILEMEMEORYSTART" << std::endl << std::flush;
      InitProfileMemory(&args);
      std::cout << "CLIENT DO_SCAMP START" << std::endl << std::flush;
      do_SCAMP(&args, std::vector<int>(), 4);

      Ta_h.clear();
      Tb_h.clear();

      *reply = ConvertArgsToReply(args);
      std::cout << "CLIENT DO_SCAMP FINISH" << std::endl << std::flush;

      std::cout << "width: " << args.max_tile_size << std::endl;

      // reply.profile_a().Swap(args.profile_a.data);
      // reply.profile_b().Swap(args.profile_b);

    } else {
      std::cout << "status not ok from server" << std::endl;

      Ta_h.clear();
      Tb_h.clear();

      // reply.mutable_profile_a()->Clear();
      // reply.mutable_profile_b()->Clear();
    }
    return ret;
  }

  SCAMPResult SCAMPCombiner(const SCAMPArgs &args) {
    SCAMPResult reply;
    ClientContext context;

    std::cout << "clietn scampcombiner" << std::endl;

    // The actual RPC.
    Status status = stub_->SCAMPCombiner(&context, args, &reply);

    std::cout << "clietn scampcombiner222222" << std::endl;

    if (status.ok()) {
      // return reply.done();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      // return -1;
      usleep(2 * 1000000);
    }
    return reply;
  }

  /*
    double Combiner(int finish, int counter, int idcnt, int arrpos, int arrsize,
    std::vector<double> &vec1)
    {
      // Data we are sending to the server.
      HelloRequest request;

      request.set_result(finish);
      request.set_reqcounter(counter);
      request.set_idcnt(idcnt);

      std::cout << "Client sent id " << idcnt << " to server result: " << finish
    << std::endl;

      //std::cout << "vecsize:" << vec1.size() << std::endl;

      for(int p = 0; p < vec1.size(); p++)
        {
          request.add_data(10);
          //std::cout << "vec1: " << vec1[p] << std::endl;
        }

      // Container for the data we expect from the server.
      HelloReply reply;

      // Context for the client. It could be used to convey extra information to
      // the server and/or tweak certain RPC behaviors.
      ClientContext context;

      vec1.clear();

      // The actual RPC.
      Status status = stub_->Combiner(&context, request, &reply);

      std::cout << "client reply.done: " << reply.done() << std::endl;

      // Act upon its status.
      if (status.ok())
        {
          return reply.done();
        }
      else
        {
          std::cout << status.error_code() << ": " << status.error_message()
                    << std::endl;
          //return "RPC failed";
          return -1;
        }
    }
  */

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char **argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  std::cout << "client start" << std::endl;

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
