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
#include <mutex>
#include <thread>
#include <vector>
#include <time.h>
#include <unistd.h>

#include <grpcpp/grpcpp.h>


#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using helloworld::HelloRequest;
using helloworld::HelloReply;
using helloworld::Greeter;

std::mutex mtx;


// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {

public:
  int counter = 0;
  int arrpos = 0;
  int combine = 0;
  int idcnt = 0;

  int reload = 0;
  int generate = 100;

  std::vector<std::vector<int>> vec1;

  static const int globarrsize = 1000;
  double globarr[globarrsize];

  GreeterServiceImpl()
  {
    counter = 0;
    arrpos = 0;
    idcnt = 0;
    for(int i = 0; i < globarrsize; i++)
      {
	globarr[i] = i;
      }
  }

private:

  void vecloop()
  {
    long int endtimer;
    int timeout = 100;
    while(true)
      {
	usleep(100*1000000);
	endtimer = time(0);

	// start lock
	{
	  std::lock_guard<std::mutex> lockGuard(mtx);
	  for(int i = 0; i < vec1.size(); i++)
	    {
	      if((vec1[i][5] == 1) && ((endtimer - vec1[i][3]) >= timeout ))
		{
		  vec1[i][5] = 0;
		}
	    }
	}
	// end lock
      }
  }
  
  void initVec(HelloReply* reply)
  {
    // remove array send, and any send, instead add # 1000 from above
    // to the array/pushback and set it all up no time yet should be -1
    // this should only initalize 1000 of the array not send anything
    // then the sendvec should actually send whats in the array to the client
    // Send entire Global Array

    // Does tiling, 1 tile per element 
    
    int status = 0;
    int endtimer = -1;

    long int timer = -1;
    int vsize = 0;
    int arrsize = 2;

    std::cout << "Server created " << globarrsize << std::endl;
    
    //{
      //std::lock_guard<std::mutex> lockGuard(mtx);

      for(int i = 0; i < globarrsize; i++)
	{
	  vec1.push_back(std::vector<int>());
      
	  vsize = vec1.size()-1;
	  
	  // Job ID
	  vec1[vsize].push_back(idcnt);
	  
	  // Job's array position to calc
	  vec1[vsize].push_back(arrpos);
	  
	  // Job's size of array portion
	  vec1[vsize].push_back(arrsize);

	  // Timer Start
	  vec1[vsize].push_back(timer);
	  // Timer End
	  vec1[vsize].push_back(endtimer);
	  
	  // Status, 0 - Ready/Waiting to run , 1 - Running, 2 - Finished
	  vec1[vsize].push_back(status);
	  
	  idcnt++;

	  if(arrpos < globarrsize)
	    {
	      arrpos = arrpos + 2;
	    }
	  else
	    {
	      arrpos = 0;
	    }
	}
      //}
  }

  
  void sendVec(HelloReply* reply)
  {
    // Send entire Global Array
    for(int p = 0; p < globarrsize; p++)
      {
        reply->add_replydata(globarr[p]);
      }

    int arrsize = 2;
    int status = 1;
    long int timer;

    // start lock
    {
      // start lock
      std::lock_guard<std::mutex> lockGuard(mtx);

      for(int i = 0; i < vec1.size(); i++)
	{
	  if(vec1[i][5] == 0)
	    {
	      // Seconds since 1970
	      timer = time(0);
	      
	      // Timer Start
	      vec1[i][3] = timer;


	      // Status, 0 - Ready/Waiting to run , 1 - Running, 2 - Finished
	      vec1[i][5] = status;
	      
	      reply->set_idcnt(vec1[i][0]);
	      reply->set_arrpos(vec1[i][1]);
	      reply->set_arrsize(vec1[i][2]);

	      
	      	      
	      std::cout << "Server sent idcnt: " << vec1[i][0] << " Start time: " << timer << std::endl;
	      
	      return;
	    }
	}
      reload++;

    }
    // end lock

    
    
  }
  
  Status SayChicken(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override
  {

    {
      // start lock
      std::lock_guard<std::mutex> lockGuard(mtx);
      // needs mutex lock ***********
      if(reload > 0)
	{
	  reload = 0;
	  initVec(reply);
	}
    }
    // end lock
    
    sendVec(reply);

    return Status::OK;
  }

  Status Combiner(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override
  {
    
    int tempid = 0;
    tempid = request->idcnt();

    std::cout << "Server Recieved job: " << tempid << "  Result: " << request->result() << std::endl;

    {
      std::lock_guard<std::mutex> lockGuard(mtx);

      for(int i = 0; i < vec1.size(); i++)
	{
	  if(vec1[i][0] == tempid)
	    {
	      if(vec1[i][5] == 1)
		{
		  combine += request->result();
		  vec1[i][5] = 2;
		  long int timer1 = time(0);
		  vec1[i][4] = timer1;
		  std::cout << "Server Found id at vec1 i: " << i << "  Result: " << request->result() << " End Time: " << timer1 << std::endl;
		  std::cout << "Combine: " << combine << std::endl;
		}
	      
	    }
	}
    }

    reply->set_done(combine);
    return Status::OK;
  }
};

void RunServer() {

  std::string server_address("0.0.0.0:30078");

  GreeterServiceImpl service;

  ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);

  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {

  std::thread vecloop();
  
  RunServer();

  return 0;
}
