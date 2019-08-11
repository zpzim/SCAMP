#pragma once
#include <cstdlib>
#include "scamp.pb.h"

// Enum describing Tile status
enum TileStatus {
  TILE_STATUS_INVALID = 0,
  TILE_STATUS_READY = 1,
  TILE_STATUS_RUNNING = 2,
  TILE_STATUS_FINISHED = 3,
  TILE_STATUS_FAILED = 4,
};

// Class describing particular tile in a Job
class DistributedTile {
 public:
  DistributedTile() : valid(false) {}
  DistributedTile(int r, int c, int64_t height, int64_t width, int id)
      : tile_id_(id),
        valid(true),
        tile_row_(r),
        tile_col_(c),
        height_(height),
        width_(width),
        status_(TILE_STATUS_READY),
        has_args_(false),
        retries_(0),
        start_time_(-1),
        end_time_(-1),
        tile_timeout_seconds_(INT_MAX) {}
  bool is_valid() const { return valid; }
  void start_time(int64_t t) { start_time_ = t; }
  void end_time(int64_t t) { end_time_ = t; }
  void status(TileStatus s) { status_ = s; }
  void tile_id(int id) { tile_id_ = id; }
  int64_t start_time() const { return start_time_; }
  int64_t end_time() const { return end_time_; }
  int tile_id() const { return tile_id_; }
  TileStatus status() const { return status_; }
  int tile_row() const { return tile_row_; }
  int tile_col() const { return tile_col_; }
  int64_t height() const { return height_; }
  int64_t width() const { return width_; }
  int64_t start_row() const { return start_row_; }
  int64_t start_col() const { return start_col_; }
  bool has_args() const { return has_args_; }
  int retries() const { return retries_; }
  void retries(int retry_count) { retries_ = retry_count; }
  void args(const SCAMPProto::SCAMPArgs &args) { args_ = args; }
  const SCAMPProto::SCAMPArgs &args() const { return args_; }
  void timeout(int timeout) { tile_timeout_seconds_ = timeout; }
  int timeout() const { return tile_timeout_seconds_; }
  const SCAMPProto::SCAMPArgs &info() const { return args_; }
  bool generate_args(const SCAMPProto::SCAMPArgs &job_args,
                     SCAMPProto::SCAMPArgs *args);
  void set_finished();
  void set_failed();

 private:
  bool valid;
  bool has_args_;
  int tile_timeout_seconds_;
  int retries_;
  int tile_row_;
  int tile_col_;
  int64_t height_;
  int64_t width_;
  int64_t start_row_;
  int64_t start_col_;

  int tile_id_;
  int64_t start_time_;
  int64_t end_time_;
  TileStatus status_;

  SCAMPProto::SCAMPArgs args_;
};
