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
#include <fstream>
#include <cmath>

#include <grpcpp/grpcpp.h>

#include "../src/SCAMP.h"
#include "../src/common.h"


#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
//using helloworld::HelloRequest;
//using helloworld::HelloReply;
using helloworld::SCAMPRequest;
using helloworld::SCAMPResult;
using helloworld::SCAMPArgs;
using helloworld::Profile;

using helloworld::Greeter;

std::mutex mtx, global_a_lock, global_b_lock;

std::vector<double> Ta_h, Tb_h;
Profile global_profile_a, global_profile_b;

//int max_tile_size = 1000000;
int max_tile_size = 30000;
int distributed_start_row = -1;
int distributed_start_col = -1;
double distance_threshold = std::numeric_limits<double>::max();
bool computing_rows = true;
bool computing_columns = true;
SCAMP::SCAMPProfileType profile_a = SCAMP::PROFILE_TYPE_1NN_INDEX;
SCAMP::SCAMPProfileType profile_b = SCAMP::PROFILE_TYPE_1NN_INDEX;
SCAMP::SCAMPProfileType profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
SCAMP::SCAMPPrecisionType precision_type = SCAMP::PRECISION_DOUBLE;
bool keep_rows_separate = false;
bool is_aligned = false;
int window = 100;
bool has_b = false;

long double num_tile_rows;
long double num_tile_cols;

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_full[i + merge_start] += to_merge[i];
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge, uint64_t index_offset) {
  for (int i = 0; i < tile_sz; ++i) {
    SCAMP::mp_entry e1, e2;
    e1.ulong = mp_full[i + merge_start];
    e2.ulong = to_merge[i];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[i + merge_start] = e2.ulong;
    }
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    if (mp_full[i + merge_start] < to_merge[i]) {
      mp_full[i + merge_start] = to_merge[i];
    }
  }
}

// TODO(zpzim): move this back into SCAMP_Operation, we shouldn't have the
// merging be functionality of the individual tile
// Merges a local result "tile_profile" with the global matrix profile
// "full_profile"
void MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                    uint64_t length, Profile *full_profile,
			      uint64_t index_start, std::mutex &lock) {
  // the entire result vector before we merge
  // TODO(zpzim): we don't have to do this, we only need to lock the specific
  // "tile row" or "tile_column" that we are updating
  std::unique_lock<std::mutex> mlock(lock);

  std::cout << "fullprofiletype: " << full_profile->type() << " position: " << position << " length: " << length  << " index start: " << index_start << std::endl;

  
  
  //switch (full_profile->type()) {
     switch (1) {
    case helloworld::PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->mutable_data()->Mutable(0)->mutable_double_value()->mutable_value()->mutable_data(),
                              position, length,
                              tile_profile->mutable_data()->Mutable(0)->mutable_double_value()->mutable_value()->mutable_data());
      return;
    case helloworld::PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(full_profile->mutable_data()->Mutable(0)->mutable_uint64_value()->mutable_value()->mutable_data(),
                             position, length,
                              tile_profile->mutable_data()->Mutable(0)->mutable_uint64_value()->mutable_value()->mutable_data());
      return;
    case helloworld::PROFILE_TYPE_1NN:
      elementwise_max<float>(full_profile->mutable_data()->Mutable(0)->mutable_float_value()->mutable_value()->mutable_data(),
                             position, length,
                              tile_profile->mutable_data()->Mutable(0)->mutable_float_value()->mutable_value()->mutable_data());
      return;
    case helloworld::PROFILE_TYPE_FREQUENCY_THRESH:
    case helloworld::PROFILE_TYPE_KNN:
    case helloworld::PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

// TODO(zpzim): move this back into SCAMP_Operation, we shouldn't have the
// merging be functionality of the individual tile
void MergeProfile(Profile *profile_a, Profile *a_tile, uint64_t col_pos, uint64_t width, std::mutex &a_lock,
			  Profile *profile_b, Profile *b_tile, uint64_t row_pos, uint64_t height, std::mutex &b_lock) {

  //MergeProfile(&global_profile_a, &tile_a, width, col_pos, global_a_lock, &global_profile_b, &tile_b, height, row_pos, global_b_lock);
  
  // Merge result
  MergeTileIntoFullProfile(a_tile, col_pos, width,
                           profile_a, row_pos, a_lock);

  std::cout << "merge profile after merge fileintofullprofile" << std::endl;
  
  // Self join
  if (true) {
    MergeTileIntoFullProfile(b_tile, row_pos, height, profile_a, col_pos, a_lock);
  }
  std::cout << "2 merge profile after merge fileintofullprofile" << std::endl;
 // else if (_info->computing_rows && _info->keep_rows_separate) {
//    MergeTileIntoFullProfile(b_tile, row_pos, height, profile_b, col_pos, b_lock);
//  }
}

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {

public:
  int counter = 0;
  int arrpos = 0;
  int combine = 0;
  int idcnt = 0;

  int reload = 1;
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

public:

  
std::ifstream &read_value(std::ifstream &s, double &d, int count) {
  std::string line;
  double parsed;

  s >> line;
  if (line.empty()) {
    if (s.peek() != EOF) {
      std::cout << "WARNING: got empty line #" << count + 1
                << " in input file\n"
                << std::endl;
    }
    d = 0;
    return s;
  }

  try {
    parsed = std::stod(line);
  } catch (std::invalid_argument e) {
    std::cout << line[0] << std::endl;
    std::cout << "FATAL ERROR: invalid argument: Could not parse line number "
              << count + 1 << " from input file.\n";
    exit(1);
  } catch (std::out_of_range e) {
    std::cout << line[0] << std::endl;
    std::cout << "FATAL ERROR: out of range: Could not parse line number "
              << count + 1 << " from input file.\n";
    exit(1);
  }
  d = parsed;
  return s;
}
  
template <class DTYPE>
  void readFile(const std::string &filename, std::vector<DTYPE> &v,
		const char *format_str) {
    std::ifstream f(filename);
    if (f.fail()) {
      std::cout << "Unable to open" << filename
		<< "for reading, please make sure it exists" << std::endl;
      exit(0);
    }
    std::cout << "Reading data from " << filename << std::endl;
    DTYPE num;
    while (read_value(f, num, v.size()) && f.peek() != EOF) {
      v.push_back(num);
    }
    std::cout << "Read " << v.size() << " values from file " << filename
	      << std::endl;
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
  
  void initVec()
  {
    static bool isinitialized = false;
    if (isinitialized) {
      return;
    }
    isinitialized = true;
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

    //std::cout << "num_tile_rows: " << num_tile_rows << std::endl;
    //std::cout << "num_tile_cols: " << num_tile_cols << std::endl;
    
    for(int r = 0; r < num_tile_rows; r++)
      {
	for(int c = 0; c < num_tile_cols; c++)
	  {
	    vec1.push_back(std::vector<int>());
      
	    vsize = vec1.size()-1;
	    
	    // Job ID
	    vec1[vsize].push_back(idcnt);
	    
	    // Job's array position to calc
	    vec1[vsize].push_back(r);
	    
	    // Job's size of array portion
	    vec1[vsize].push_back(c);
	    
	    // Timer Start
	    vec1[vsize].push_back(timer);
	    // Timer End
	    vec1[vsize].push_back(endtimer);
	    
	    // Status, 0 - Ready/Waiting to run , 1 - Running, 2 - Finished
	    vec1[vsize].push_back(status);
	    
	    idcnt++;
	    
	  }
      }

    //print
    /*
    for(int r = 0; r < num_tile_rows*num_tile_cols; r++)
      {
	std::cout << "vec1 ids: " << vec1[r][0] << std::endl;
	std::cout << "vec1 row: " << vec1[r][1] << std::endl;
	std::cout << "vec1 col: " << vec1[r][2] << std::endl;
	std::cout << "vec1 status: " << vec1[r][5] << std::endl;
	std::cout << std::endl;
      }
    */
  }
  
  
  bool sendVec(SCAMPArgs* reply)
  {
    // Fix decide if self join so B timeseries doesn't go or if B timeseries data should go and not self join?
    //???????????????????????????????????????????

    std::cout << "Start Send Vec" << std::endl;

    int arrsize = 2;
    int status = 1;
    int tileAsize = 0;
    int tileBsize = 0;
    long int timer;

    tileAsize = ceil(Ta_h.size()/num_tile_cols);
    tileBsize = ceil(Tb_h.size()/num_tile_rows);

    std::cout << "tileAsize: " << tileAsize << std::endl;
    std::cout << "tileBsize: " << tileBsize << std::endl;
    
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
	      
	      reply->set_job_id(vec1[i][0]); // this is idcnt
	      //reply->set_tile_row(vec1[i][1]);
	      //reply->set_tile_col(vec1[i][2]);

	      uint64_t start_col = (vec1[i][1] * tileAsize);
	      uint64_t end_col = (((vec1[i][1]+1) * tileAsize) + window -1);
	     
	      if(end_col > Ta_h.size())
		{
		  end_col = Ta_h.size();
		}
	      
	      //std::cout << "a start: " << start << std::endl;
	      //std::cout << "a end: " << end << std::endl;
	      //std::cout << "size ta_h: " << Ta_h.size() << std::endl;
	      
	      for(uint64_t i = start_col; i < end_col; i++)
		{
		  reply->add_timeseries_a(Ta_h[i]);
		}
	      reply->set_timeseries_size_a(end_col - start_col);
	      //std::cout << "finished a" << std::endl;

	      uint64_t start_row = (vec1[i][2] * tileAsize);
	      uint64_t end_row = (((vec1[i][2]+1) * tileAsize) + window - 1);

	      if(end_row > Tb_h.size())
		{
		  end_row = Tb_h.size();
		}
	      
	      //std::cout << "b start: " << start << std::endl;
	      //std::cout << "b end: " << end << std::endl;
	      //std::cout << "size tb_h: " << Tb_h.size() << std::endl;
	      
	      for(uint64_t i = start_row; i < end_row; i++)
		//for(int i = 0; i < Tb_h.size(); i++)
		{
		  reply->add_timeseries_b(Tb_h[i]);
		}
	      
	      // fix

	      reply->set_timeseries_size_b(end_row - start_row);
	      
	      reply->set_max_tile_size(max_tile_size);
	      reply->set_distributed_start_row(start_row);
	      reply->set_distributed_start_col(start_col);
	      reply->set_distance_threshold(distance_threshold);
	      reply->set_computing_columns(computing_columns);
	      reply->set_computing_rows(computing_rows);
	      //?????????????????????????????????
	      //reply->set_profile_a(profile_a);
	      //reply->set_profile_b(profile_b);
	      //reply->set_precision_type(precision_type);
	      //reply->set_profile_type(profile_type);
	      reply->set_keep_rows_separate(keep_rows_separate);
	      reply->set_is_aligned(is_aligned);
	      reply->set_window(window);
	      reply->set_has_b(has_b);

	      
	      	      
	      std::cout << "Server sent idcnt: " << vec1[i][0] << " Start time: " << timer << std::endl;
	      
	      return true;
	    }
	}
      //std::cout << "reload set to 1****" << std::endl;
      //reload++;

      return false;
      
    }
    // end lock

    
    
  }

  Status RequestSCAMPWork(ServerContext* context, const SCAMPRequest* request,
                  SCAMPArgs* reply) override
  {

    {
      // start lock
      std::lock_guard<std::mutex> lockGuard(mtx);
      // needs mutex lock ***********
      std::cout << "requestscampwork in" << std::endl;
      /*if(reload > 0)
	{
	  std::cout << "reloading 1" << std::endl;
	  reload = 0;*/
	  initVec();
	  //}
    }
    // end lock

    std::cout << "before send vec call" << std::endl;
    

    if(sendVec(reply))
      {
	return Status::OK;
      }
    else
      {
	return Status::CANCELLED;
      }

  }


  Status SCAMPCombiner (ServerContext* context, const SCAMPArgs* request,
                  SCAMPResult* reply) override
  {
    std::cout << "SERVER SCAMPCOMBINER" << std::endl;
    uint64_t height = request->timeseries_size_b();
    uint64_t width = request->timeseries_size_a();
    uint64_t row_pos = request->distributed_start_row();
    uint64_t col_pos = request->distributed_start_col();
    helloworld::Profile tile_a = request->profile_a();
    helloworld::Profile tile_b = request->profile_b();

    std::cout << "SERVER SCAMPCOMBINER 2" << std::endl;
    std::cout << "height: " << height << " width: " << width << " row_pos: " << row_pos << " col_pos: " << col_pos << std::endl;

    
    printf("merging\n");
    MergeProfile(&global_profile_a, &tile_a, col_pos, width, global_a_lock, &global_profile_b, &tile_b, row_pos, height, global_b_lock);
    printf("merging after\n");
    std::cout << "after merging print: " << global_profile_a.mutable_data()->Add()->mutable_uint64_value()->mutable_value() << std::endl;
    
    //std::vector<double> outVector;
    //std::transform(global_profile_a.data[0].cbegin(), global_profile_a.data[0].cend(), outVector.begin(), [](const double& in){return in;});

    //double chicken = global_profile_a.data[0].double_value.begin();
    
    //std::cout << "global_profile_a: " << global_profile_a.data[0].double_value.begin() << std::endl;
    
    printf("merge done\n");
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
    
  GreeterServiceImpl obj;
  obj.readFile<double>("test/SampleInput/randomlist128K.txt", Ta_h, "%lf");
  obj.readFile<double>("test/SampleInput/randomlist128K.txt", Tb_h, "%lf");
  std::cout << "array a size: " << Ta_h.size() << std::endl;
  std::cout << "array b size: " << Tb_h.size() << std::endl;
  //readFile<double>("test/SampleInput/randomlist8K.txt", Ta_h, "%lf");
  SCAMP::mp_entry initializer;
  initializer.floats[0] = -2;
  initializer.ints[1] = 0;
  global_profile_a.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(Ta_h.size()-window+1, initializer.ulong);
  global_profile_b.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(Tb_h.size()-window+1, initializer.ulong);

  num_tile_rows = ceil((Tb_h.size() - window + 1) /
		       static_cast<double>(max_tile_size));
  num_tile_cols = ceil((Ta_h.size() - window + 1) /
		       static_cast<double>(max_tile_size));
  
  std::thread vecloop();
  
  RunServer();

  return 0;
}
