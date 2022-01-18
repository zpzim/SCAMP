#pragma once
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "common/common.h"
#include "common/profile.h"
#include "qt_helper.h"

namespace SCAMP {

class Tile {
 private:
  // Per worker input vectors
  std::unique_ptr<double, std::function<void(double *)>> T_A_dev_, T_B_dev_,
      QT_dev_, means_A_, means_B_, norms_A_, norms_B_, df_A_, df_B_, dg_A_,
      dg_B_, scratchpad_;

  std::unique_ptr<float, std::function<void(float *)>> thresholds_A_,
      thresholds_B_;
  // Per worker output vectors (device)
  DeviceProfile profile_a_tile_dev_, profile_b_tile_dev_;

  // Length of the output vector on the device
  // Set by the kernel when the profile can have a variable length
  unsigned long long int *profile_a_dev_length_;
  unsigned long long int *profile_b_dev_length_;

  // Vector of matches used for ALL_NEIGHBORS joins
  std::vector<SCAMPmatch> _matches_local;

  // Per worker output vectors (host)
  Profile profile_a_tile_, profile_b_tile_;

  // Helper for FFT computation
  std::shared_ptr<qt_compute_helper> scratch_;

  // Tile-specific variables describing the current tile
  size_t current_tile_width_;
  size_t current_tile_height_;
  size_t current_tile_col_;
  size_t current_tile_row_;
  // True if this tile has nan inputs.
  bool has_nan_input_;

  const OpInfo *info_;
  ExecInfo exec_info_;

  // Private member functions which perform joins
  SCAMPError_t do_self_join_full();
  SCAMPError_t do_self_join_half();
  SCAMPError_t do_ab_join_full();

  void init();
  void init_cuda();
  void init_cpu();
  void free_cuda();
  void free_cpu();
  void Memset(void *destination, char value, size_t bytes);
  void Memcopy(void *destination, const void *source, size_t bytes,
               bool from_tile);
  Profile AllocProfile(SCAMPProfileType t, uint64_t size);
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length);
  void SortMatches(SCAMPmatch *matches, uint64_t len);

  // Gets the profile length computed by the kernel when there can be a variable
  // length
  std::pair<int64_t, int64_t> get_profile_dims_from_device();

 public:
  Tile(const OpInfo *info);
  Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id);
  ~Tile();
#ifdef _HAS_CUDA_
  cudaStream_t get_stream() { return exec_info_.stream; }
  cudaDeviceProp get_dev_props() { return exec_info_.dev_props; }
#endif

  int get_cuda_id() const {
#ifdef _HAS_CUDA_
    return exec_info_.cuda_id;
#else
    return -1;
#endif
  }
  SCAMPArchitecture get_arch() const { return exec_info_.arch; }
  size_t get_tile_width() const { return current_tile_width_; }
  size_t get_tile_height() const { return current_tile_height_; }
  size_t get_tile_row() const { return current_tile_row_; }
  size_t get_tile_col() const { return current_tile_col_; }
  bool has_nan_input() const { return has_nan_input_; }
  const OpInfo *info() const { return info_; }
  void *profile_a() { return profile_a_tile_dev_.at(info_->profile_type); };
  void *profile_b() { return profile_b_tile_dev_.at(info_->profile_type); };
  double *QT() { return QT_dev_.get(); }
  const double *dfa() const { return df_A_.get(); }
  const double *dfb() const { return df_B_.get(); }
  const double *dga() const { return dg_A_.get(); }
  const double *dgb() const { return dg_B_.get(); }
  const double *normsa() const { return norms_A_.get(); }
  const double *normsb() const { return norms_B_.get(); }
  const float *thresholds_A() const { return thresholds_A_.get(); }
  const float *thresholds_B() const { return thresholds_B_.get(); }
  unsigned long long int *get_mutable_a_dev_length() {
    return profile_a_dev_length_;
  }
  unsigned long long int *get_mutable_b_dev_length() {
    return profile_b_dev_length_;
  }

  std::pair<int, int> get_exclusion_for_ab_join(bool upper_tile);
  std::pair<int, int> get_exclusion_for_self_join(bool upper_tile);
  void set_tile_col(size_t col) { current_tile_col_ = col; }
  void set_tile_row(size_t row) { current_tile_row_ = row; }
  void set_tile_height(size_t height) { current_tile_height_ = height; }
  void set_tile_width(size_t width) { current_tile_width_ = width; }
  bool MergeProfile(Profile *profile_a, Profile *profile_b);
  void Sync();

  // Initializes the precomputed statistics required by the current tile
  void InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b);

  void InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b,
                 const CombinedStats &ab);

  // Initializes the ouptut vector with the current best-so-far profile
  SCAMPError_t InitProfile(Profile *profile_a, Profile *profile_b);

  // Initializes the time series for the current tile
  void InitTimeseries(const std::vector<double> &Ta_h,
                      const std::vector<double> &Tb_h);

  // Executes a pre-initialized tile
  SCAMPError_t execute(SCAMPTileType t);
};

}  // namespace SCAMP
