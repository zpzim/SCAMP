#include "tile.h"
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

SCAMPError_t SCAMP_Tile::do_self_join_full() {
/*
  SCAMPError_t error;

  if (window_size > tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (window_size > tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  error =
      fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_upper(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp_type,
      opt_args, profile_type, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error =
      fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_lower(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, props, fp_type,
      opt_args, profile_type, _stream);
  if (error != SCAMP_NO_ERROR) {
    printf("SCAMP error\n");
    return error;
  }
*/
  return SCAMP_NO_ERROR;

}

SCAMPError_t SCAMP_Tile::do_self_join_half() {
  SCAMPError_t error;

  if (_worker->_mp_window > _worker->get_tile_width()) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (_worker->_mp_window > _worker->get_tile_height()) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  printf("Computing QT\n");
  error =
      _worker->_scratch->compute_QT(_worker->_QT_dev, _worker->_T_A_dev, _worker->_T_B_dev, _worker->_means_B, _worker->_stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error = kernel_self_join_upper(
      _worker->_QT_dev, _worker->_df_A, _worker->_df_B, _worker->_dg_A, _worker->_dg_B, _worker->_norms_A, _worker->_norms_B, &_worker->_profile_a_tile_dev,
      &_worker->_profile_b_tile_dev, _worker->_mp_window, _worker->_current_tile_width - _worker->_mp_window + 1,
      _worker->_current_tile_height - _worker->_mp_window + 1, _worker->_current_tile_col, _worker->_current_tile_row, _worker->_dev_props, fp_type,
      opt_args, _worker->_profile_type, _worker->_stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Tile::do_ab_join_full() {
/*
  SCAMPError_t error;
  if (window_size > tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (window_size > tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  error =
      fft_info->compute_QT(QT_scratch, timeseries_A, timeseries_B, means_B, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  error = kernel_ab_join_upper(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, global_start_A,
      global_start_B, aligned_ab_join, props, fp_type, full_join, opt_args,
      profile_type, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  error =
      fft_info->compute_QT(QT_scratch, timeseries_B, timeseries_A, means_A, _stream);
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  error = kernel_ab_join_lower(
      QT_scratch, df_A, df_B, dg_A, dg_B, norms_A, norms_B, profile_A,
      profile_B, window_size, tile_width - window_size + 1,
      tile_height - window_size + 1, tile_start_A, tile_start_B, global_start_A,
      global_start_B, aligned_ab_join, props, fp_type, full_join, opt_args,
      profile_type, _stream);

  if (error != SCAMP_NO_ERROR) {
    printf("SCAMP error\n");
    return error;
  }

*/
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
