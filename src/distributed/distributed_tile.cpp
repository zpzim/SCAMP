#include "distributed_tile.h"

#include "scamp.pb.h"
#include "utils.h"

bool DistributedTile::generate_args(const SCAMPProto::SCAMPArgs &job_args,
                                    SCAMPProto::SCAMPArgs *args) {
  start_col_ = (tile_col_ * job_args.distributed_tile_size());
  uint64_t end_col = start_col_ + width_;

  start_row_ = (tile_row_ * job_args.distributed_tile_size());
  uint64_t end_row = start_row_ + height_;

  args->set_timeseries_size_a(end_col - start_col_ + job_args.window() - 1);
  args->set_timeseries_size_b(end_row - start_row_ + job_args.window() - 1);
  args->set_distributed_start_row(start_row_);
  args->set_distributed_start_col(start_col_);
  args->set_max_tile_size(job_args.max_tile_size());
  args->set_distance_threshold(job_args.distance_threshold());
  args->set_profile_type(job_args.profile_type());
  args->set_precision_type(job_args.precision_type());
  args->set_window(job_args.window());
  args->set_is_aligned(job_args.is_aligned());

  // If the full job is a self join
  if (!job_args.has_b()) {
    args->set_computing_columns(true);
    args->set_computing_rows(true);
    // Check if we are on the diagonal, as we can save time by perfoming
    // smaller self-joins
    if (tile_row_ == tile_col_) {
      args->set_has_b(false);
      args->set_keep_rows_separate(job_args.keep_rows_separate());
    } else {
      args->set_has_b(true);
      args->set_keep_rows_separate(true);
      args->set_is_aligned(true);
    }
  } else {
    args->set_has_b(true);
    args->set_computing_columns(true);
    args->set_computing_rows(job_args.keep_rows_separate());
    args->set_keep_rows_separate(job_args.keep_rows_separate());
  }

  args->mutable_profile_a()->set_type(job_args.profile_type());
  args->mutable_profile_b()->set_type(job_args.profile_type());
  args->set_tile_id(tile_id_);

  // Make a copy of the arguments passed to this tile, but do not store the
  // actual time series as this would require more space than necessary
  this->args(*args);
  has_args_ = true;
  // Set the timeseries values in the tile arguments
  for (uint64_t i = start_col_; i < end_col + job_args.window() - 1; i++) {
    args->add_timeseries_a(job_args.timeseries_a()[i]);
  }
  if (job_args.has_b()) {
    for (uint64_t i = start_row_; i < end_row + job_args.window() - 1; i++) {
      args->add_timeseries_b(job_args.timeseries_b()[i]);
    }
  } else {
    // TODO(zpzim): we don't always need to set this
    for (uint64_t i = start_row_; i < end_row + job_args.window() - 1; i++) {
      args->add_timeseries_b(job_args.timeseries_a()[i]);
    }
  }

  return true;
}

// Sets tile to be complete
void DistributedTile::set_finished() {
  status_ = TILE_STATUS_FINISHED;
  end_time_ = get_current_time();
}

// Sets tile to failed
void DistributedTile::set_failed() {
  status_ = TILE_STATUS_FAILED;
  end_time_ = get_current_time();
}
