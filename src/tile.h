#pragma once
#include <memory>
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

class SCAMP_Tile {
 private:
  SCAMPTileType type;
  const double *timeseries_A;
  const double *timeseries_B;
  const double *df_A;
  const double *df_B;
  const double *dg_A;
  const double *dg_B;
  const double *means_A;
  const double *means_B;
  const double *norms_A;
  const double *norms_B;
  std::shared_ptr<fft_precompute_helper> fft_info;
  double *QT_scratch;
  uint64_t *profile_A;
  uint64_t *profile_B;

  size_t global_start_A;
  size_t global_start_B;
  size_t tile_start_A;
  size_t tile_start_B;
  size_t tile_height;
  size_t tile_width;
  const size_t window_size;
  bool full_join;
  const FPtype fp_type;
  const cudaDeviceProp props;

  SCAMPError_t do_self_join_full(cudaStream_t s);
  SCAMPError_t do_self_join_half(cudaStream_t s);
  SCAMPError_t do_ab_join_full(cudaStream_t s);
  SCAMPError_t do_ab_join_upper(cudaStream_t s);
  SCAMPError_t do_ab_join_lower(cudaStream_t s);

 public:
  SCAMP_Tile(SCAMPTileType t, const double *ts_A, const double *ts_B,
             const double *dfA, const double *dfB, const double *dgA,
             const double *dgB, const double *normA, const double *normB,
             const double *meansA, const double *meansB, double *QT,
             uint64_t *profileA, uint64_t *profileB, size_t start_A,
             size_t start_B, size_t g_start_A, size_t g_start_B, size_t height,
             size_t width, size_t m,
             std::shared_ptr<fft_precompute_helper> scratch,
             const cudaDeviceProp &prop, FPtype fp_t)
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
        full_join(false) {}
  SCAMPError_t execute(cudaStream_t s) {
    SCAMPError_t error;
    switch (type) {
      case SELF_JOIN_FULL_TILE:
        error = do_self_join_full(s);
        break;
      case SELF_JOIN_UPPER_TRIANGULAR:
        error = do_self_join_half(s);
        break;
      case AB_JOIN_FULL_TILE:
        error = do_ab_join_full(s);
        break;
      case AB_FULL_JOIN_FULL_TILE:
        full_join = true;
        error = do_ab_join_full(s);
        break;
      default:
        error = SCAMP_TILE_ILLEGAL_TYPE;
        break;
    }
    return error;
  }
};

}  // namespace SCAMP
