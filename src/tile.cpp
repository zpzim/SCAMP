#include "tile.h"
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

SCAMPError_t SCAMP_Tile::do_self_join_full(cudaStream_t s) {
  SCAMPError_t error;

  if (window_size > tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (window_size > tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  error =
      fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_upper(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp_type,
      opt_args, profile_type, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error =
      fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_lower(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp_type,
      opt_args, profile_type, s);
  if (error != SCAMP_NO_ERROR) {
    printf("SCAMP error\n");
    return error;
  }

  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Tile::do_self_join_half(cudaStream_t s) {
  SCAMPError_t error;

  if (window_size > tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (window_size > tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  error =
      fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_upper(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp_type,
      opt_args, profile_type, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Tile::do_ab_join_full(cudaStream_t s) {
  SCAMPError_t error;
  if (window_size > tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (window_size > tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  error =
      fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  /*
    error = kernel_ab_join_upper(
        QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A,
        norms_B, profile_A, profile_B, window_size, tile_width - window_size +
    1, tile_height - window_size + 1, tile_start_A, tile_start_B,
    global_start_A, global_start_B, props, fp_type, full_join, thresh, s);
  */
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error =
      fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, s);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  /*
    error = kernel_ab_join_lower(
        QT_scratch, timeseries_A, timeseries_B, df_A, df_B, dg_A, dg_B, norms_A,
        norms_B, profile_A, profile_B, window_size, tile_width - window_size +
    1, tile_height - window_size + 1, tile_start_A, tile_start_B,
    global_start_A, global_start_B, props, fp_type, full_join, thresh, s);
  */
  if (error != SCAMP_NO_ERROR) {
    printf("SCAMP error\n");
    return error;
  }

  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
