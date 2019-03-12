#pragma once
#include <memory>
#include "SCAMPWorker.h"
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

class SCAMP_Tile {
 private:
  SCAMPTileType type;
  Worker *_worker;
//  const double *timeseries_A;
//  const double *timeseries_B;

//  const double *df_A;
//  const double *df_B;
///  const double *dg_A;
//  const double *dg_B;
//  const double *means_A;
//  const double *means_B;
//  const double *norms_A;
//  const double *norms_B;
//  std::shared_ptr<fft_precompute_helper> fft_info;
//  double *QT_scratch;
//  DeviceProfile *profile_A;
//  DeviceProfile *profile_B;
  OptionalArgs opt_args;
  int64_t global_start_A;
  int64_t global_start_B;
//  size_t tile_start_A;
//  size_t tile_start_B;
//  size_t tile_height;
//  size_t tile_width;
  bool aligned_ab_join;
//  const size_t window_size;
  bool full_join;
  double thresh;
  const SCAMPPrecisionType fp_type;
//  const SCAMPProfileType profile_type;
#ifdef _HAS_CUDA_
//  const cudaDeviceProp props;
//  cudaStream_t _stream;
#endif

 public:
  SCAMP_Tile(SCAMPTileType t, Worker *worker, size_t g_start_A, size_t g_start_B, bool aligned, SCAMPPrecisionType fp_t, OptionalArgs _opt_args)
      : type(t),
        _worker(worker),
        //timeseries_A(worker->_T_A_dev),
        //timeseries_B(worker->_T_B_dev),
        //df_A(worker->_df_A),
        //df_B(worker->_df_B),
        //means_A(worker->_means_A),
        //means_B(worker->_means_B),
        //dg_A(worker->_dg_A),
        //dg_B(worker->_dg_B),
        //norms_A(worker->_norms_A),
        //norms_B(worker->_norms_B),
        //QT_scratch(worker->_QT_dev),
        //profile_A(&worker->_profile_a_tile_dev),
        //profile_B(&worker->_profile_b_tile_dev),
        //tile_start_A(worker->_current_tile_col),
        //tile_start_B(worker->_current_tile_row),
        global_start_A(g_start_A),
        global_start_B(g_start_B),
        //tile_height(worker->_current_tile_height),
        //tile_width(worker->_current_tile_width),
        //fft_info(worker->_scratch),
        //window_size(worker->_mp_window),
        //props(worker->_dev_props),
        fp_type(fp_t),
        full_join(false),
        aligned_ab_join(aligned),
        //profile_type(worker->_profile_type),
        opt_args(_opt_args) {}
 /*     
  SCAMP_Tile(SCAMPTileType t, const double *ts_A, const double *ts_B,
             const double *dfA, const double *dfB, const double *dgA,
             const double *dgB, const double *normA, const double *normB,
             const double *meansA, const double *meansB, double *QT,
             DeviceProfile *profileA, DeviceProfile *profileB, size_t start_A,
             int64_t start_B, int64_t g_start_A, size_t g_start_B, bool aligned,
             size_t height, size_t width, size_t m,
             std::shared_ptr<fft_precompute_helper> scratch,
             const cudaDeviceProp &prop, SCAMPPrecisionType fp_t,
             SCAMPProfileType profile_type_, OptionalArgs opt_args_)
      : type(t),
        timeseries_A(ts_A),
        timeseries_B(ts_B),
        df_A(dfA),
        df_B(dfB),
        means_A(meansA),
        means_B(meansB),
        dg_A(dgA),
        dg_B(dgB),
        norms_A(normA),
        norms_B(normB),
        QT_scratch(QT),
        profile_A(profileA),
        profile_B(profileB),
        tile_start_A(start_A),
        tile_start_B(start_B),
        global_start_A(g_start_A),
        global_start_B(g_start_B),
        tile_height(height),
        tile_width(width),
        fft_info(scratch),
        window_size(m),
        props(prop),
        fp_type(fp_t),
        full_join(false),
        aligned_ab_join(aligned),
        profile_type(profile_type_),
        opt_args(opt_args_) {}
*/
  SCAMPError_t do_self_join_full();
  SCAMPError_t do_self_join_half();
  SCAMPError_t do_ab_join_full();
  SCAMPError_t execute() {
    SCAMPError_t error;
    switch (type) {
      case SELF_JOIN_FULL_TILE:
        error = do_self_join_full();
        break;
      case SELF_JOIN_UPPER_TRIANGULAR:
        error = do_self_join_half();
        break;
      case AB_JOIN_FULL_TILE:
        error = do_ab_join_full();
        break;
      case AB_FULL_JOIN_FULL_TILE:
        full_join = true;
        error = do_ab_join_full();
        break;
      default:
        error = SCAMP_TILE_ILLEGAL_TYPE;
        break;
    }
    return error;
  }
};

}  // namespace SCAMP
