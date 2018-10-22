#pragma once
#include <memory>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

/*

using std::tuple<SCAMPTileType, FPtype, DeviceInfo, uint64_t, uint64_t,
uint64_t, uint64_t, uint64_t, uint64_t, bool, bool> = SCAMPTileParamList;

struct SCAMPStream {
public:
  cudaStream_t get_gpu_stream() { return *cuda_stream; }
  void get_cpu_stream()
private:
  // Cuda GPU
  cudaStream_t *cuda_stream;
  // CPU
  // TODO: TO IMPLEMENT
}


class DeviceInfo {
public:
  bool is_gpu_target { return arch_type_ == ARCH_TYPE_GPU; }
  bool is_cpu_target { return arch_type_ == ARCH_TYPE_CPU; }
  SCAMPArchSubtype get_subtype { return arch_subtype_ };
private:
  SCAMPArchType arch_type_;
  SCAMPArchSubType arch_subtype_;
  cudaDeviceProp gpu_properties;
}

class SCAMPTile {
public:
  SCAMPTile(const SCAMPTileParamList& params)
    : tile_type_(std::get<0>(params)),
      fp_type_(std::get<1>(params)),
      device_info_(std::get<2>(params)),
      computing_rows_(std::get<3>(params)),
      computing_columns_(std::get<4>(params)),
      tile_height_(std::get<5>(params)),
      tile_width_(std::get<6>(params)),
      tile_start_height_(std::get<7>(params)),
      tile_start_width_(std::get<8>(params)),
      distributed_tile_start_height_(std::get<9>(params)),
      distributed_tile_start_width_(std::get<10>(params))
  {};
  SCAMPTile(SCAMPTileParamList&& params)
    : tile_type_(std::get<0>(params)),
      fp_type_(std::get<1>(params)),
      device_info_(std::get<2>(params)),
      computing_rows_(std::get<3>(params)),
      computing_columns_(std::get<4>(params)),
      tile_height_(std::get<5>(params)),
      tile_width_(std::get<6>(params)),
      tile_start_height_(std::get<7>(params)),
      tile_start_width_(std::get<8>(params)),
      distributed_tile_start_height_(std::get<9>(params)),
      distributed_tile_start_width_(std::get<10>(params))
  {};
  virtual SCAMPError_t execute(cudaStream_t s) {
    SCAMPError_t error;
    switch (type_) {
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
protected:
  virtual SCAMPError_t do_self_join_full(SCAMPStream *s) = 0;
  virtual SCAMPError_t do_self_join_half(SCAMPStream *s) = 0;
  virtual SCAMPError_t do_ab_join_full(SCAMPStream *s) = 0;
  const SCAMPTileType tile_type_;
  const FPtype fp_type_;
  const DeviceInfo device_info_;
  const uint64_t tile_height_;
  const uint64_t tile_width_;
  const uint64_t tile_start_height_;
  const uint64_t tile_start_width_;
  const uint64_t distributed_tile_start_height_;
  const uint64_t distributed_tile_start_width_;
  const bool computing_rows_;
  const bool computing_columns_;
};

class SCAMPInputs {
  MatrixProfileInputs a;
  MatrixProfileOutputs b;


};

class SCAMPProfile {
  uint64_t profile
  vector<pair<std::vector<>, T*>> profiles
  vector<SCAMPProfileType> profile_types  // parallel vector to profiles,
describes the type of profile


template < typename T >
struct MatrixProfileInputs : public SCAMPInputs {
    const T *timeseries;
    const T *df;
    const T *dg;
    const T *means;
    const T *norms;
};

template < typename T>
struct MatrixProfileOutputs : public SCAMPOutputs {
  T *profile;
};

template <typename T_OUT >
class SCAMPMatrixProfileTileBase : public SCAMPTile {
public:
  SCAMPMatrixProfileTile(SCAMPTileParamList&& tile_params, const SCAMPInputs& a,
const SCAMPInputs& b, std::shared_ptr<fft_precompute_helper> fft_info, double
*scratchpad, uint64_t window_size) : SCAMPTile(tile_params), a_(a), b_(b),
out_(out), fft_info_(fft_info), window_size(window_size_) {} protected:
  MatrixProfileInputs<double> a_;
  MatrixProfileInputs<double> b_;
  std::shared_ptr<fft_precompute_helper> fft_info_;
  double *scratchpad_;
  const uint64_t window_size_;
  virtual SCAMPError_t do_self_join_full(SCAMPStream *s);
  virtual SCAMPError_t do_self_join_half(SCAMPStream *s);
  virtual SCAMPError_t do_ab_join_full(SCAMPStream *s);
};

template <typename T_OUT>
class SCAMPSumMPTile : public SCAMPMatrixProfileTile {


}
*/

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
  double *profile_A;
  double *profile_B;
  double thresh;
  size_t global_start_A;
  size_t global_start_B;
  size_t tile_start_A;
  size_t tile_start_B;
  size_t tile_height;
  size_t tile_width;
  const size_t window_size;
  bool full_join;
  const SCAMPPrecisionType fp_type;
  const cudaDeviceProp props;

 public:
  SCAMP_Tile(SCAMPTileType t, const double *ts_A, const double *ts_B,
             const double *dfA, const double *dfB, const double *dgA,
             const double *dgB, const double *normA, const double *normB,
             const double *meansA, const double *meansB, double *QT,
             double *profileA, double *profileB, size_t start_A, size_t start_B,
             size_t g_start_A, size_t g_start_B, size_t height, size_t width,
             size_t m, std::shared_ptr<fft_precompute_helper> scratch,
             const cudaDeviceProp &prop, SCAMPPrecisionType fp_t, double th)
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
        thresh(th) {}
  SCAMPError_t do_self_join_full(cudaStream_t s);
  SCAMPError_t do_self_join_half(cudaStream_t s);
  SCAMPError_t do_ab_join_full(cudaStream_t s);
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
