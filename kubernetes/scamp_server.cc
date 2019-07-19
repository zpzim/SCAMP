#include <time.h>
#include <unistd.h>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "../src/SCAMP.h"
#include "../src/common.h"
#include "scamp.grpc.pb.h"
//chadd
//#include "google.golang.org/grpc/reflection"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using SCAMPProto::Profile;
using SCAMPProto::SCAMPArgs;
using SCAMPProto::SCAMPRequest;
using SCAMPProto::SCAMPResult;
using SCAMPProto::SCAMPStatus;

using SCAMPProto::SCAMPService;

class Job;

std::vector<Job> jobVec;
std::mutex jobVecLock;

constexpr int DISTRIBUTED_TILE_SIZE = 500000;

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
  // std::cout << "Reading data from " << filename << std::endl;
  DTYPE num;
  while (read_value(f, num, v.size()) && f.peek() != EOF) {
    v.push_back(num);
  }
  // std::cout << "Read " << v.size() << " values from file " << filename
  //          << std::endl;
}

//chad
/*
void check_time_out()
{
  for(int i = 0; i < jobVec.size(); i++)
  {
     if(!jobVec[i].job_done())
     {
        for(int k = 0; k < jobVec[i].
     }
  }
}
*/

enum TileStatus {
  TILE_STATUS_INVALID = 0,
  TILE_STATUS_READY = 1,
  TILE_STATUS_RUNNING = 2,
  TILE_STATUS_FINISHED = 3,
};

class Tile {
 public:
  Tile() : valid(false) {}
  Tile(int r, int c, int id)
      : tile_id_(id),
        valid(true),
        tile_row_(r),
        tile_col_(c),
        status_(TILE_STATUS_READY),
        start_time_(-1),
        end_time_(-1) {}
  bool is_valid() { return valid; }
  void start_time(int64_t t) { start_time_ = t; }
  void end_time(int64_t t) { end_time_ = t; }
  void status(TileStatus s) { status_ = s; }
  void tile_id(int id) { tile_id_ = id; }
  int64_t start_time() { return start_time_; }
  int64_t end_time() { return end_time_; }
  int tile_id() { return tile_id_; }
  TileStatus status() { return status_; }
  int tile_row() { return tile_row_; }
  int tile_col() { return tile_col_; }
  void args(const SCAMPArgs &args) { args_ = args; }
  const SCAMPArgs &args() const { return args_; }

 private:
  bool valid;
  int tile_row_;
  int tile_col_;
  int tile_id_;
  int64_t start_time_;
  int64_t end_time_;
  TileStatus status_;
  // Contains a copy of the arguments used by the tile. But not the data.
  SCAMPArgs args_;
};

// Thread safe queue to hold tiles to be executed
class TileQueue {
 public:
  size_t size() { return queue_.size(); }

  bool empty() {
    // std::lock_guard<std::mutex> lockGuard(mutex_);
    return queue_.empty();
  }

  // Pop an element from the queue, if the queue is already empty, return the
  // sentinel !valid which indicates that there was no data in the queue
  Tile *pop() {
    if (queue_.empty()) {
      return nullptr;
    }

    Tile *item = queue_.front();
    queue_.pop();
    return item;
  }

  void push(Tile *item) { queue_.push(item); }

 private:
  std::queue<Tile *> queue_;
};

class Job {
 public:
  Job(SCAMPArgs args, int id) : job_id(id), job_args(args), tile_counter(0) {
    Init();
  };
  /*
    void print_state() {
      std::cout << "ready queue size: " << ready_queue.size() << " running queue
    size: " << running.size() << " finished size: " << finished.size() <<
    std::endl;
    }
  */
  void set_finished(int tile_id) {
    tiles[tile_id].status(TILE_STATUS_FINISHED);
    tiles[tile_id].end_time(time(0));
  }

  bool job_done() {
    for (auto elem : tiles) {
      if (elem.second.status() != TILE_STATUS_FINISHED) {
        return false;
      }
    }
    return true;
  }

  bool fetch_ready_tile(SCAMPArgs *args) {
    if (ready_queue.empty()) {
      std::cout << "Ready queue was empty for job " << job_id << std::endl;
      return false;
    }

    Tile *tile = ready_queue.pop();
    if (tile == nullptr) {
      return false;
    }

    args->set_tile_id(tile->tile_id());
    args->set_job_id(job_id);

    uint64_t tileAsize = job_args.timeseries_a().size() / tile_cols;
    uint64_t tileBsize = job_args.timeseries_b().size() / tile_rows;

    uint64_t start_col = (tile->tile_col() * tileAsize);
    uint64_t end_col =
        (((tile->tile_col() + 1) * tileAsize) + job_args.window() - 1);

    if (end_col > job_args.timeseries_a().size()) {
      end_col = job_args.timeseries_a().size();
    }

    uint64_t start_row = (tile->tile_row() * tileBsize);
    uint64_t end_row =
        (((tile->tile_row() + 1) * tileBsize) + job_args.window() - 1);

    if (end_row > job_args.timeseries_b().size()) {
      end_row = job_args.timeseries_b().size();
    }

    args->set_timeseries_size_a(end_col - start_col);
    args->set_timeseries_size_b(end_row - start_row);
    args->set_distributed_start_row(start_row);
    args->set_distributed_start_col(start_col);
    args->set_max_tile_size(job_args.max_tile_size());
    args->set_is_aligned(true);
    args->set_distance_threshold(job_args.distance_threshold());
    args->set_profile_type(job_args.profile_type());
    args->set_precision_type(job_args.precision_type());
    args->set_window(job_args.window());

    if (!job_args.has_b()) {
      args->set_computing_columns(true);
      args->set_computing_rows(true);
      if (tile->tile_row() == tile->tile_col()) {
        args->set_has_b(false);
        args->set_keep_rows_separate(job_args.keep_rows_separate());
      } else {
        args->set_has_b(true);
        args->set_keep_rows_separate(true);
      }
    } else {
      args->set_computing_columns(true);
      args->set_computing_rows(job_args.keep_rows_separate());
      args->set_keep_rows_separate(job_args.keep_rows_separate());
    }

    args->mutable_profile_a()->set_type(job_args.profile_type());
    args->mutable_profile_b()->set_type(job_args.profile_type());

    // Make a copy of the arguments passed to this tile, but do not store the
    // actual time series as this would require more space than necessary
    tile->args(*args);

    for (uint64_t i = start_col; i < end_col; i++) {
      args->add_timeseries_a(job_args.timeseries_a()[i]);
    }
    for (uint64_t i = start_row; i < end_row; i++) {
      args->add_timeseries_b(job_args.timeseries_b()[i]);
    }
    // Timer Start
    tile->start_time(time(0));
    tile->status(TILE_STATUS_RUNNING);
    return true;
  }
  SCAMPArgs *get_job_args() { return &job_args; }
  const SCAMPArgs &get_tile_args(int tile_id) { return tiles[tile_id].args(); }

 //chad
 /* 
  bool check_queue()
  {
    
  }*/

 private:
  void Init() {
    tile_cols = ceil((job_args.timeseries_a().size() - job_args.window() + 1) /
                     static_cast<double>(DISTRIBUTED_TILE_SIZE));
    if (job_args.has_b()) {
      tile_rows =
          ceil((job_args.timeseries_b().size() - job_args.window() + 1) /
               static_cast<double>(DISTRIBUTED_TILE_SIZE));
    } else {
      tile_rows = tile_cols;
    }

    for (int r = 0; r < tile_rows; r++) {
      for (int c = 0; c < tile_cols; c++) {
        tiles.emplace(tile_counter, Tile(r, c, tile_counter));
        ready_queue.push(&tiles[tile_counter]);
        tile_counter++;
      }
    }
  }
  int job_id;
  int tile_counter;
  int tile_rows;
  int tile_cols;
  std::unordered_map<int, Tile> tiles;
  TileQueue ready_queue;

  // This is better as a set (key is tile id)
  SCAMPArgs job_args;
};

void createTestJob() {
  int window = 100;
  SCAMP::mp_entry initializer;
  initializer.floats[0] = -2;
  initializer.ints[1] = 0;

  std::vector<double> Ta_h;
  readFile<double>("../test/SampleInput/randomlist128K.txt", Ta_h, "%lf");

  SCAMPProto::SCAMPArgs args;
  *args.mutable_timeseries_a() = std::move(
      google::protobuf::RepeatedField<double>(Ta_h.begin(), Ta_h.end()));

  // FIXME need if for has_b then line below needs b in it???
  *args.mutable_timeseries_b() = std::move(
      google::protobuf::RepeatedField<double>(Ta_h.begin(), Ta_h.end()));

  args.mutable_profile_a()->set_type(SCAMPProto::PROFILE_TYPE_1NN_INDEX);
  args.mutable_profile_a()
      ->mutable_data()
      ->Add()
      ->mutable_uint64_value()
      ->mutable_value()
      ->Resize(Ta_h.size() - window + 1, initializer.ulong);
  /*
    args.mutable_profile_b()->set_type(SCAMPProto::PROFILE_TYPE_1NN_INDEX);
    args.mutable_profile_b()->mutable_data()
        ->Add()
        ->mutable_uint64_value()
        ->mutable_value()
        ->Resize(Tb_h.size() - window + 1, initializer.ulong);
  */

  args.set_max_tile_size(1000000);
  args.set_distributed_start_row(-1);
  args.set_distributed_start_col(-1);
  // FIXME
  args.set_distance_threshold(std::numeric_limits<double>::max());
  args.set_computing_rows(true);
  args.set_computing_columns(true);
  args.set_profile_type(SCAMPProto::PROFILE_TYPE_1NN_INDEX);
  args.set_precision_type(SCAMPProto::PRECISION_DOUBLE);
  args.set_keep_rows_separate(false);
  args.set_is_aligned(false);
  args.set_window(window);
  args.set_has_b(false);
  std::ofstream filestream("./testproto.proto");
  filestream << args.DebugString();
  //if(args.SerializeToOstream(&filestream)) {
  //  std::cout << "Wrote proto to file" << std::endl;
  //}
  filestream.close();
  //jobVec.emplace_back(args, 0);
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
  std::cout << "fullprofiletype: " << full_profile->type()
            << " position: " << position << " length: " << length
            << " index start: " << index_start << std::endl;

  switch (full_profile->type()) {
    case SCAMPProto::PROFILE_TYPE_SUM_THRESH:
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
    case SCAMPProto::PROFILE_TYPE_1NN_INDEX:
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
                                    ->mutable_data(),
                                index_start);
      return;
    case SCAMPProto::PROFILE_TYPE_1NN:
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
    case SCAMPProto::PROFILE_TYPE_FREQUENCY_THRESH:
    case SCAMPProto::PROFILE_TYPE_KNN:
    case SCAMPProto::PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

void MergeProfile(const SCAMPArgs &tile_args, SCAMPArgs *job_args,
                  Profile *a_tile, uint64_t col_pos, uint64_t width,
                  Profile *b_tile, uint64_t row_pos, uint64_t height) {
  // Merge result
  MergeTileIntoFullProfile(a_tile, col_pos, width,
                           job_args->mutable_profile_a(), row_pos);
  if (tile_args.keep_rows_separate()) {
    if (job_args->computing_rows() && job_args->keep_rows_separate()) {
      MergeTileIntoFullProfile(b_tile, row_pos, height,
                               job_args->mutable_profile_b(), col_pos);
    } else if (!job_args->has_b()) {
      MergeTileIntoFullProfile(b_tile, row_pos, height,
                               job_args->mutable_profile_a(), col_pos);
    }
  }
}

// Logic and data behind the server's behavior.
class SCAMPServiceImpl final : public SCAMPService::Service {
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

  SCAMPServiceImpl() {
    counter = 0;
    arrpos = 0;
    idcnt = 0;
    for (int i = 0; i < globarrsize; i++) {
      globarr[i] = i;
    }
  }

 public:
 private:
  // TODO(chad): The following RPCS need to be defined so that we can issue and
  // complete jobs on the server You will need to define the new protos
  // associated with SCAMPStatus and SCAMPResult in scamp.proto These RPCS will
  // be called by another command line program on a different machine

  /*
  // Takes a SCAMPArgs proto and tries to create a SCAMP job and add it to the
  jobVec
  // Returns a job id in SCAMPStatus if we create a job
  // Returns failure state in SCAMPStatus if we fail to create a job
  */
  Status IssueNewJob(ServerContext *context, const SCAMPProto::SCAMPArgs
  *job_args, SCAMPStatus *status) override {

  std::lock_guard<std::mutex> lockGuard(jobVecLock);

  uint64_t cur_job_id = jobVec.size();
  jobVec.emplace_back(*job_args, cur_job_id);
  
  status->set_status(true);
  status->set_job_id(cur_job_id);
  return Status::OK;
  
  }
  

  // Takes a job_id and returns the status of the job associated with that ID
  Status CheckJobStatus(ServerContext *context, const SCAMPProto::SCAMPJobID *SCAMP_job_id, SCAMPStatus
  *status) override {
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    if (SCAMP_job_id->job_id() >= jobVec.size() || SCAMP_job_id->job_id() < 0) {
      status->set_status(false);
      return Status::CANCELLED;
    }
    status->set_status(jobVec[SCAMP_job_id->job_id()].job_done());
    return Status::OK;
  }

  
  // Takes a job id and returns the completed work associated with that id if
  // the job has been completed
  // Otherwise returns a null result
  Status FetchJobResult(ServerContext *context, const SCAMPProto::SCAMPJobID *SCAMP_job_id, SCAMPProto::SCAMPWork *job_result) override {
    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    if (SCAMP_job_id->job_id() >= jobVec.size() || SCAMP_job_id->job_id() < 0) {
      job_result->set_valid(false);
      return Status::CANCELLED;
    }
    if(jobVec[SCAMP_job_id->job_id()].job_done()){
    
	*job_result->mutable_args() = *jobVec[SCAMP_job_id->job_id()].get_job_args();
        job_result->set_valid(true);
    }else {
        job_result->set_valid(false);
    }
    return Status::OK;
  }
  

  Status RequestSCAMPWork(ServerContext *context, const SCAMPRequest *request,
                          SCAMPProto::SCAMPWork *reply) override {
    std::cout << "Work requested from server" << std::endl;
    SCAMPArgs *args = reply->mutable_args();
    std::lock_guard<std::mutex> lockGuard(jobVecLock);

    for (int i = 0; i < jobVec.size(); i++) {
      std::cout << "Checking for ready tile for job " << i << std::endl;
      if (jobVec[i].fetch_ready_tile(args)) {
        std::cout << "Tile fetched from job " << i << std::endl;
        reply->set_valid(true);
        return Status::OK;
      }
    }
    reply->set_valid(false);
    return Status::OK;
  }

  Status SCAMPCombiner(ServerContext *context, const SCAMPArgs *request,
                       SCAMPResult *reply) override {
    uint64_t height = request->timeseries_size_b();
    uint64_t width = request->timeseries_size_a();
    uint64_t row_pos = request->distributed_start_row();
    uint64_t col_pos = request->distributed_start_col();
    SCAMPProto::Profile tile_a = request->profile_a();
    SCAMPProto::Profile tile_b = request->profile_b();

    int job_id = request->job_id();
    int tile_id = request->tile_id();
    std::cout << "Combining tile " << tile_id << " for job " << job_id
              << std::endl;
    std::cout << "start_row: " << row_pos << " start_col: " << col_pos
              << std::endl;
    std::cout << "height: " << height << " width: " << width << std::endl;

    std::lock_guard<std::mutex> lockGuard(jobVecLock);
    MergeProfile(jobVec[job_id].get_tile_args(tile_id),
                 jobVec[job_id].get_job_args(), &tile_a, col_pos, width,
                 &tile_b, row_pos, height);

    //chad
    //jobVec[job_id].check_queue();

    jobVec[job_id].set_finished(tile_id);
    if (jobVec[job_id].job_done()) {
      for (auto elem : jobVec[job_id]
                           .get_job_args()
                           ->profile_a()
                           .data()
                           .Get(0)
                           .uint64_value()
                           .value()) {
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

  SCAMPServiceImpl service;

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
  // TODO: move this into an asynchronous rpc which appends a new job to jobVec
  createTestJob();

  //std::thread check_time_out();

  RunServer();
  return 0;
}
