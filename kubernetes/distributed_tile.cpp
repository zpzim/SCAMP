#include "distributed_tile.h"

#include "../src/scamp_utils.h"
#include "scamp.pb.h"

bool DistributedTile::generate_args(const SCAMPProto::SCAMPArgs &job_args,
                                    int rows, int columns,
                                    SCAMPProto::SCAMPArgs *args) {
  const SCAMPProto::SCAMPInfo &job_info = job_args.info();
  SCAMPProto::SCAMPInfo *info = args->mutable_info();
  uint64_t Asize = job_info.timeseries_size_a();
  uint64_t Bsize = job_info.has_b() ? job_info.timeseries_size_b() : Asize;

  uint64_t tileAsize = Asize / columns;
  uint64_t tileBsize = Bsize / rows;

  uint64_t start_col = (tile_col_ * tileAsize);
  uint64_t end_col = (((tile_col_ + 1) * tileAsize) + job_info.window() - 1);

  if (end_col > Asize) {
    end_col = Asize;
  }

  uint64_t start_row = (tile_row_ * tileBsize);
  uint64_t end_row = (((tile_row_ + 1) * tileBsize) + job_info.window() - 1);

  if (end_row > Bsize) {
    end_row = Bsize;
  }

  info->set_timeseries_size_a(end_col - start_col);
  info->set_timeseries_size_b(end_row - start_row);
  info->set_distributed_start_row(start_row);
  info->set_distributed_start_col(start_col);
  info->set_max_tile_size(job_info.max_tile_size());
  info->set_distance_threshold(job_info.distance_threshold());
  info->set_profile_type(job_info.profile_type());
  info->set_precision_type(job_info.precision_type());
  info->set_window(job_info.window());
  info->set_is_aligned(job_info.is_aligned());

  // If the full job is a self join
  if (!job_info.has_b()) {
    info->set_computing_columns(true);
    info->set_computing_rows(true);
    // Check if we are on the diagonal, as we can save time by perfoming
    // smaller self-joins
    if (tile_row_ == tile_col_) {
      info->set_has_b(false);
      info->set_keep_rows_separate(job_info.keep_rows_separate());
    } else {
      info->set_has_b(true);
      info->set_keep_rows_separate(true);
      info->set_is_aligned(true);
    }
  } else {
    info->set_has_b(true);
    info->set_computing_columns(true);
    info->set_computing_rows(job_info.keep_rows_separate());
    info->set_keep_rows_separate(job_info.keep_rows_separate());
  }

  args->mutable_profile_a()->set_type(job_info.profile_type());
  args->mutable_profile_b()->set_type(job_info.profile_type());

  // Make a copy of the arguments passed to this tile, but do not store the
  // actual time series as this would require more space than necessary
  this->info(*info);

  if (job_info.use_file_io()) {
    std::vector<double> A;
    readFile(job_info.timeseries_a_file_path(), A, "%lf");
    if (A.size() != job_info.timeseries_size_a()) {
      return false;
    }
    *args->mutable_timeseries_a() = {A.begin(), A.end()};
    if (job_info.has_b()) {
      std::vector<double> B;
      readFile(job_info.timeseries_b_file_path(), B, "%lf");
      if (B.size() != job_info.timeseries_size_b()) {
        return false;
      }
      *args->mutable_timeseries_b() = {B.begin(), B.end()};
    }
  } else {
    for (uint64_t i = start_col; i < end_col; i++) {
      args->add_timeseries_a(job_args.timeseries_a()[i]);
    }
    if (job_info.has_b()) {
      for (uint64_t i = start_row; i < end_row; i++) {
        args->add_timeseries_b(job_args.timeseries_b()[i]);
      }
    } else {
      for (uint64_t i = start_row; i < end_row; i++) {
        args->add_timeseries_b(job_args.timeseries_a()[i]);
      }
    }
  }
  info->set_tile_id(tile_id_);
  return true;
}
