#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"

namespace SCAMP {

class Tile {
 private:
  // Architecture
  SCAMPArchitecture _arch;

  // GPU id (if applicable)
  int _cuda_id;

  // Per worker input vectors
  std::unique_ptr<double, std::function<void(double *)>> _T_A_dev, _T_B_dev,
      _QT_dev, _means_A, _means_B, _norms_A, _norms_B, _df_A, _df_B, _dg_A,
      _dg_B, _scratchpad;

  // Per worker output vectors (device)
  DeviceProfile _profile_a_tile_dev, _profile_b_tile_dev;

  // Per worker output vectors (host)
  Profile _profile_a_tile, _profile_b_tile;

  // Helper for FFT computation
  std::shared_ptr<fft_precompute_helper> _scratch;

  // Tile-specific variables describing the current tile
  size_t _current_tile_width;
  size_t _current_tile_height;
  size_t _current_tile_col;
  size_t _current_tile_row;

  const OpInfo *_info;
#ifdef _HAS_CUDA_
  // Cuda stream and device properties associated with this worker
  cudaStream_t _stream;
  cudaDeviceProp _dev_props;
#endif

  // Private member functions which perform joins
  SCAMPError_t do_self_join_full();
  SCAMPError_t do_self_join_half();
  SCAMPError_t do_ab_join_full();

  void init();
  void init_cuda();
  void init_cpu();
  void free_cuda();
  void free_cpu();
  Profile AllocProfile(SCAMPProfileType t, uint64_t size);
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length);
  void MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                uint64_t length, Profile *full_profile,
                                uint64_t index_start, std::mutex &lock);

 public:
  Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id);
  ~Tile();
#ifdef _HAS_CUDA_
  cudaStream_t get_stream() { return _stream; }
#endif

  SCAMPArchitecture get_arch() { return _arch; }
  int get_cuda_id() { return _cuda_id; }
  size_t get_tile_width() { return _current_tile_width; }
  size_t get_tile_height() { return _current_tile_height; }
  size_t get_tile_row() { return _current_tile_row; }
  size_t get_tile_col() { return _current_tile_col; }
  void set_tile_col(size_t col) { _current_tile_col = col; }
  void set_tile_row(size_t row) { _current_tile_row = row; }
  void set_tile_height(size_t height) { _current_tile_height = height; }
  void set_tile_width(size_t width) { _current_tile_width = width; }
  void MergeProfile(Profile *profile_a, std::mutex &a_lock, Profile *profile_b,
                    std::mutex &b_lock);
  void Sync();

  // Initializes the precomputed statistics required by the current tile
  void InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b);

  // Initializes the ouptut vector with the current best-so-far profile
  SCAMPError_t InitProfile(Profile *profile_a, Profile *profile_b);

  // Initializes the time series for the current tile
  void InitTimeseries(const google::protobuf::RepeatedField<double> &Ta_h,
                      const google::protobuf::RepeatedField<double> &Tb_h);

  // Executes a pre-initialized tile
  SCAMPError_t execute(SCAMPTileType t);
};

}  // namespace SCAMP
