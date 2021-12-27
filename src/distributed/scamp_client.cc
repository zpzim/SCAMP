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

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>

#include "scamp_worker.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

int main(int argc, char **argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  // std::cout << "client start" << std::endl;

  char *port;
  char *ip;

  port = getenv("SCAMP_SERVER_SERVICE_PORT");
  ip = getenv("SCAMP_SERVER_SERVICE_HOST");

  std::string newip, newport;

  if (ip != nullptr && port != nullptr) {
    newip = ip;
    newport = port;
  } else {
    newip = "localhost";
    newport = "30078";
  }

  std::string good = newip + ":" + newport;

  std::cout << "Using addr: " << good << std::endl;

  grpc::ChannelArguments ch_args;

  // Do not limit input size
  ch_args.SetMaxReceiveMessageSize(-1);

  SCAMPWorker worker(grpc::CreateCustomChannel(
      good, grpc::InsecureChannelCredentials(), ch_args));

  bool failed = worker.run();

  return failed;
}
