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

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>



#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using helloworld::HelloRequest;
using helloworld::HelloReply;
using helloworld::Greeter;

class GreeterClient {
 public:

  int randnum;

  GreeterClient()
  {
    randnum = 0;
    srand (time(NULL));
    randnum = rand();
  }
  
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}
  
  double SayChicken(const std::string& user, int &counter, int &idcnt, int &arrpos, int &arrsize, std::vector<double> & vec1) {

    // Data we are sending to the server.
    HelloRequest request;
    
    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayChicken(&context, request, &reply);

    // Act upon its status.
    if (status.ok())
      {

	double x = 0;
	double temp;

	idcnt = reply.idcnt();
	arrpos = reply.arrpos();
	arrsize = reply.arrsize();

	std::cout << "Client Recieved id: " << idcnt << " arrpos: " << arrpos << " arrsize: " << arrsize << std::endl;
	
	for(int i = arrpos; i < (arrpos + reply.arrsize()); i++)
	  {
	    temp = reply.replydata()[i];
	    x += temp;
	    
	    vec1.push_back(temp);
	  }

	usleep(5*1000000);
	return x;
      }
    else
      {
	std::cout << status.error_code() << ": " << status.error_message()
		  << std::endl;
	//return "RPC failed";
	return -1;
      }
  }


  double Combiner(int finish, int counter, int idcnt, int arrpos, int arrsize, std::vector<double> &vec1)
  {
    // Data we are sending to the server.
    HelloRequest request;
    
    request.set_result(finish);
    request.set_reqcounter(counter);
    request.set_idcnt(idcnt);

    std::cout << "Client sent id " << idcnt << " to server result: " << finish << std::endl;
    
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
  
 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char** argv)
{

  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  std::cout << "client start" << std::endl;

  char* port;
  char* ip;

  port = getenv ("SERVERVEC_SERVICE_PORT");
  ip = getenv ("SERVERVEC_SERVICE_HOST");

  if(ip != NULL)
    {
      std::cout << "ip: " << ip << std::endl;
      std::cout << "port: " << port << std::endl;
    }

  std::string good = std::string(ip) + ":" + std::string(port);

  //std::cout << "goodid: " << good << std::endl;
  
  GreeterClient greeter(grpc::CreateChannel(good, grpc::InsecureChannelCredentials()));
  
  std::string user("worldchicken");
  std::vector<double> vec1;
  
  int counter = 0;
  int idcnt = 0;
  int arrpos = 0;
  int arrsize = 0;

  while(true)
    {
      double reply1 = greeter.SayChicken(user, counter, idcnt, arrpos, arrsize, vec1);
      double reply2 = greeter.Combiner(reply1, counter, idcnt, arrpos, arrsize, vec1);

      std::cout << "reply1chickenmethodGreeter received: " << reply1 << std::endl;
      std::cout << "reply2Combiner received: " << reply2 << std::endl;
    }
  
  return 0;
}
