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
  DistributedTile(int r, int c, int id)
      : tile_id_(id),
        valid(true),
        tile_row_(r),
        tile_col_(c),
        status_(TILE_STATUS_READY),
        retries_(0),
        start_time_(-1),
        end_time_(-1),
        tile_timeout_seconds_(INT_MAX) {}
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
  int retries() { return retries_; }
  int retries(int retry_count) { retries_ = retry_count; }
  void info(const SCAMPProto::SCAMPInfo &info) { info_ = info; }
  void timeout(int timeout) { tile_timeout_seconds_ = timeout; }
  int timeout() { return tile_timeout_seconds_; }
  const SCAMPProto::SCAMPInfo &info() const { return info_; }
  bool generate_args(const SCAMPProto::SCAMPArgs &job_args, int rows,
                     int columns, SCAMPProto::SCAMPArgs *args);

 private:
  bool valid;
  int tile_timeout_seconds_;
  int retries_;
  int tile_row_;
  int tile_col_;
  int tile_id_;
  int64_t start_time_;
  int64_t end_time_;
  TileStatus status_;

  SCAMPProto::SCAMPInfo info_;
};
