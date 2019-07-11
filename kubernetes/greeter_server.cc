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

#include <time.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

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
using helloworld::Profile;
using helloworld::SCAMPArgs;
using helloworld::SCAMPRequest;
using helloworld::SCAMPResult;

using helloworld::Greeter;

class Job;

std::vector<Job> jobVec;
std::mutex jobVecLock;

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
  //std::cout << "Reading data from " << filename << std::endl;
  DTYPE num;
  while (read_value(f, num, v.size()) && f.peek() != EOF) {
    v.push_back(num);
  }
  //std::cout << "Read " << v.size() << " values from file " << filename
  //          << std::endl;
}

enum TileStatus {
  TILE_STATUS_INVALID = 0,
  TILE_STATUS_READY = 1,
  TILE_STATUS_RUNNING = 2,
  TILE_STATUS_FINISHED = 3,
};

class Tile {
public:
  Tile() : valid(false) {}
  Tile(int r, int c, int id) : tile_id_(id), valid(true), tile_row_(r), tile_col_(c), status_(TILE_STATUS_READY), start_time_(-1), end_time_(-1) {}
  bool is_valid()
  {
    return valid;
  }
  void start_time(int64_t t) { start_time_ = t; }
  void end_time(int64_t t) { end_time_ = t; }
  void status(TileStatus s) { status_ = s; }
  void tile_id(int id) { tile_id_ = id; }
  int64_t start_time() { return start_time_; }
  int64_t end_time() { return end_time_; }
  int tile_id() { return tile_id_; }
  TileStatus status() { return status_; }
  int tile_row() { return tile_row_;}
  int tile_col() { return tile_col_;}

  
private:
  bool valid;
  int tile_row_;
  int tile_col_;
  int tile_id_;
  int64_t start_time_;
  int64_t end_time_;
  TileStatus status_;
  SCAMPArgs args;
};


// Thread safe queue to hold tiles to be executed
class TileQueue {
 public:

  size_t size() {
    return queue_.size();
  }

  bool empty() {
    //std::lock_guard<std::mutex> lockGuard(mutex_);
    return queue_.empty();
  }

  // Pop an element from the queue, if the queue is already empty, return the
  // sentinel !valid which indicates that there was no data in the queue
  Tile pop() {
    Tile item = Tile();
    if (!queue_.empty()) {
      item = queue_.front();
      queue_.pop();
    }
    return item;
  }

  void push(const Tile& item) {
    queue_.push(item);
  }

  void push(Tile&& item) {
    queue_.push(std::move(item));
  }  
  
 private:
  std::queue<Tile> queue_;
};

class Job {
public:
  
  Job(SCAMPArgs args, int id) : job_id(id), job_args(args), tile_counter(0) { Init(); };

  void print_state() {
    std::cout << "ready queue size: " << ready_queue.size() << " running queue size: " << running.size() << " finished size: " << finished.size() << std::endl;
  }
  
  void set_finished(int tile_id) {
    for (int i = 0; i < running.size(); ++i) {
      if (running[i].tile_id() == tile_id) {
	running[i].end_time(time(0));
	finished.push_back(running[i]);
      	running.erase(running.begin() + i);
	return;
      }
    }
  }
  
  bool job_done() {
    return running.empty() && ready_queue.empty();
  }

  bool fetch_ready_tile(helloworld::SCAMPArgs *args) {
    if(ready_queue.empty()) {
      return false;
    }
    
    Tile tile = ready_queue.pop();
    if (!tile.is_valid()) {
      // This should not happen
      return false;
    }
	    
    // Seconds since 1970
    long int timer = time(0);
      
    // Timer Start
    tile.start_time(timer);
    
    tile.status(TILE_STATUS_RUNNING);
      
    args->set_tile_id(tile.tile_id());
    args->set_job_id(job_id); 
      
    uint64_t tileAsize = job_args.timeseries_a().size() / tile_cols;
    uint64_t tileBsize = job_args.timeseries_b().size() / tile_rows;

    uint64_t start_col = (tile.tile_col() * tileAsize);
    uint64_t end_col = (((tile.tile_col() + 1) * tileAsize) + job_args.window() - 1);
      
    if (end_col > job_args.timeseries_a().size()) {
	end_col = job_args.timeseries_a().size();
    }

      
    for (uint64_t i = start_col; i < end_col; i++) {
	args->add_timeseries_a(job_args.timeseries_a()[i]);
    }

    args->set_timeseries_size_a(end_col - start_col);

    uint64_t start_row = (tile.tile_row() * tileBsize);
    uint64_t end_row = (((tile.tile_row() + 1) * tileBsize) + job_args.window() - 1);

    if (end_row > job_args.timeseries_b().size()) {
	end_row = job_args.timeseries_b().size();
    }

    for (uint64_t i = start_row; i < end_row; i++) {
	args->add_timeseries_b(job_args.timeseries_b()[i]);
    }
    args->set_timeseries_size_b(end_row - start_row);
      
    // TODO Lots of tile specific logic

    args->set_distributed_start_row(start_row);
    args->set_distributed_start_col(start_col);
    args->set_max_tile_size(job_args.max_tile_size());
    args->set_is_aligned(true);
    args->set_distance_threshold(job_args.distance_threshold());
    args->set_profile_type(job_args.profile_type());
    args->set_precision_type(job_args.precision_type());
    args->set_window(job_args.window());
    args->set_keep_rows_separate(job_args.keep_rows_separate());

    // If self join
    if (!job_args.has_b()) {
      args->set_computing_columns(true);
      args->set_computing_rows(true);
      if (tile.tile_row() == tile.tile_col()) {
        args->set_has_b(false);
      } else {
        args->set_has_b(true);
      }
    } else {
      args->set_computing_columns(true);
      args->set_computing_rows(job_args.keep_rows_separate());
    }
      
    args->mutable_profile_a()->set_type(job_args.profile_type());
    args->mutable_profile_b()->set_type(job_args.profile_type());

    running.push_back(tile);
    return true;
  }
  SCAMPArgs* get_job_args() { return &job_args; }
  
private:
  void Init()
  {
    tile_rows =
      ceil((job_args.timeseries_b().size() - job_args.window() + 1) / static_cast<double>(job_args.max_tile_size()));
    tile_cols =
      ceil((job_args.timeseries_a().size() - job_args.window() + 1) / static_cast<double>(job_args.max_tile_size()));
    
    for (int r = 0; r < tile_rows; r++) {
      for (int c = 0; c < tile_cols; c++) {
	Tile tile(r,c,tile_counter++);
	ready_queue.push(tile);
      }
    }
  }
  int job_id;
  int tile_counter;
  int tile_rows;
  int tile_cols;
  TileQueue ready_queue;

  // This is better as a set (key is tile id)
  std::vector<Tile> running;
  std::vector<Tile> finished;
  SCAMPArgs job_args;
};


void createTestJob()
{
  int window = 100;
  SCAMP::mp_entry initializer;
  initializer.floats[0] = -2;
  initializer.ints[1] = 0;

  std::vector<double> Ta_h, Tb_h;
  readFile<double>("test/SampleInput/randomlist128K.txt", Ta_h, "%lf");
  readFile<double>("test/SampleInput/randomlist128K.txt", Tb_h, "%lf");
  //std::cout << "array a size: " << Ta_h.size() << std::endl;
  //std::cout << "array b size: " << Tb_h.size() << std::endl;
  
  helloworld::SCAMPArgs args;

  *args.mutable_timeseries_a() = std::move(google::protobuf::RepeatedField<double>(Ta_h.begin(), Ta_h.end()));
  *args.mutable_timeseries_b() = std::move(google::protobuf::RepeatedField<double>(Tb_h.begin(), Tb_h.end()));

  args.mutable_profile_a()->set_type(helloworld::PROFILE_TYPE_1NN_INDEX);
  args.mutable_profile_a()->mutable_data()
      ->Add()
      ->mutable_uint64_value()
      ->mutable_value()
      ->Resize(Ta_h.size() - window + 1, initializer.ulong);
  args.mutable_profile_b()->set_type(helloworld::PROFILE_TYPE_1NN_INDEX);
  args.mutable_profile_b()->mutable_data()
      ->Add()
      ->mutable_uint64_value()
      ->mutable_value()
      ->Resize(Tb_h.size() - window + 1, initializer.ulong);

  args.set_max_tile_size(1000000);
  args.set_distributed_start_row(-1);
  args.set_distributed_start_col(-1);
  //FIXME
  args.set_distance_threshold(std::numeric_limits<double>::max());
  args.set_computing_rows(true);
  args.set_computing_columns(true);
  args.set_profile_type(helloworld::PROFILE_TYPE_1NN_INDEX);
  args.set_precision_type(helloworld::PRECISION_DOUBLE);
  args.set_keep_rows_separate(false);
  args.set_is_aligned(false);
  args.set_window(window);
  args.set_has_b(false);
  jobVec.emplace_back(args, 0);
}

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
                              uint64_t index_start) {

  /*
  std::cout << "fullprofiletype: " << full_profile->type()
            << " position: " << position << " length: " << length
            << " index start: " << index_start << std::endl;
  */

  // switch (full_profile->type()) {
  //FIXME
  switch (1) {
    case helloworld::PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data(),
                              position, length,
                              tile_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data());
      return;
    case helloworld::PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(full_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data());
      return;
    case helloworld::PROFILE_TYPE_1NN:
      elementwise_max<float>(full_profile->mutable_data()
                                 ->Mutable(0)
                                 ->mutable_float_value()
                                 ->mutable_value()
                                 ->mutable_data(),
                             position, length,
                             tile_profile->mutable_data()
                                 ->Mutable(0)
                                 ->mutable_float_value()
                                 ->mutable_value()
                                 ->mutable_data());
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
void MergeProfile(Profile *profile_a, Profile *a_tile, uint64_t col_pos,
                  uint64_t width, Profile *profile_b,
                  Profile *b_tile, uint64_t row_pos, uint64_t height
                  ) {

  //FIXME this all correct now?
  
  // MergeProfile(&global_profile_a, &tile_a, width, col_pos, global_a_lock,
  // &global_profile_b, &tile_b, height, row_pos, global_b_lock);

  // Merge result
  MergeTileIntoFullProfile(a_tile, col_pos, width, profile_a, row_pos);

  //std::cout << "merge profile after merge fileintofullprofile" << std::endl;

  // Self join
  // if (true) {
  //  MergeTileIntoFullProfile(b_tile, row_pos, height, profile_a, col_pos,
  //  a_lock);
  //}
  
  //std::cout << "2 merge profile after merge fileintofullprofile" << std::endl;

  // else if (_info->computing_rows && _info->keep_rows_separate) {
  //    MergeTileIntoFullProfile(b_tile, row_pos, height, profile_b, col_pos,
  //    b_lock);
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

  GreeterServiceImpl() {
    counter = 0;
    arrpos = 0;
    idcnt = 0;
    for (int i = 0; i < globarrsize; i++) {
      globarr[i] = i;
    }
  }

 public:

  
 private:

  Status RequestSCAMPWork(ServerContext *context, const SCAMPRequest *request,
                          helloworld::SCAMPWork *reply) override {

    SCAMPArgs *args = reply->mutable_args();
    std::lock_guard<std::mutex> lockGuard(jobVecLock);

    for(int i = 0; i < jobVec.size(); i++)
    {
      //std::cout << "Fetching tile" << std::endl;
      if (jobVec[i].fetch_ready_tile(args))
	{
	  reply->set_valid(true);
	  //std::cout << "Request work finished" << std::endl;
	  return Status::OK;
	}
    }
    reply->set_valid(false);
    return Status::OK;
  }
  
  Status SCAMPCombiner(ServerContext *context, const SCAMPArgs *request,
                       SCAMPResult *reply) override {

    //std::cout << "SERVER SCAMPCOMBINER" << std::endl;
    
    uint64_t height = request->timeseries_size_b();
    uint64_t width = request->timeseries_size_a();
    uint64_t row_pos = request->distributed_start_row();
    uint64_t col_pos = request->distributed_start_col();
    helloworld::Profile tile_a = request->profile_a();
    helloworld::Profile tile_b = request->profile_b();

    /*
    std::cout << "SERVER SCAMPCOMBINER 2" << std::endl;
    std::cout << "height: " << height << " width: " << width
              << " row_pos: " << row_pos << " col_pos: " << col_pos
              << std::endl;
    */

    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    int job_id = request->job_id();
    int tile_id = request->tile_id();
    
    //std::cout << "job: " << job_id << " tile: " << tile_id << std::endl;

    MergeProfile(jobVec[job_id].get_job_args()->mutable_profile_a(), &tile_a, col_pos, width,
                 jobVec[job_id].get_job_args()->mutable_profile_b(), &tile_b, row_pos, height);
    
    jobVec[job_id].set_finished(tile_id);
    jobVec[job_id].print_state();
    if (jobVec[job_id].job_done()){ 
      for (auto elem : jobVec[job_id].get_job_args()->profile_a().data().Get(0).uint64_value().value()) {
	SCAMP::mp_entry e;
	e.ulong = elem;
	printf("%lf\n", e.floats[0]);
      }
    }
    std::cout << "Finished Merging" << std::endl;
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

int main(int argc, char **argv) {

  //GreeterServiceImpl initVect();
      
  //std::cout << "before create jobs" << std::endl;
  
  createTestJob();

  //std::cout << "after create jobs" << std::endl;
  
  RunServer();

  return 0;
}
