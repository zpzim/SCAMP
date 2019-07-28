#include "distributed_tile.h"

#include "../src/scamp_utils.h"
#include "scamp.pb.h"

bool DistributedTile::generate_args(const SCAMPProto::SCAMPArgs &job_args,
                                    int rows, int columns,
                                    SCAMPProto::SCAMPArgs *args) {
  uint64_t Asize = job_args.timeseries_a().size();
  uint64_t Bsize = job_args.has_b() ? job_args.timeseries_b().size() : Asize;

  uint64_t tileAsize = Asize / columns;
  uint64_t tileBsize = Bsize / rows;

  uint64_t start_col = (tile_col_ * tileAsize);
  uint64_t end_col = (((tile_col_ + 1) * tileAsize) + job_args.window() - 1);

  if (end_col > Asize) {
    end_col = Asize;
  }

  uint64_t start_row = (tile_row_ * tileBsize);
  uint64_t end_row = (((tile_row_ + 1) * tileBsize) + job_args.window() - 1);

  if (end_row > Bsize) {
    end_row = Bsize;
  }

  args->set_timeseries_size_a(end_col - start_col);
  args->set_timeseries_size_b(end_row - start_row);
  args->set_distributed_start_row(start_row);
  args->set_distributed_start_col(start_col);
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

  // Set the timeseries values in the tile arguments
  for (uint64_t i = start_col; i < end_col; i++) {
    args->add_timeseries_a(job_args.timeseries_a()[i]);
  }
  if (job_args.has_b()) {
    for (uint64_t i = start_row; i < end_row; i++) {
      args->add_timeseries_b(job_args.timeseries_b()[i]);
    }
  } else {
    // TODO(zpzim): we don't always need to set this
    for (uint64_t i = start_row; i < end_row; i++) {
      args->add_timeseries_b(job_args.timeseries_a()[i]);
    }
  }

  return true;
}
