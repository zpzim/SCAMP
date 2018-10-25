#pragma once
#include <cuda.h>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
using std::list;
using std::pair;
using std::unordered_map;
using std::vector;

namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices);

class SCAMP_Operation {
 private:
  unordered_map<int, double *> T_A_dev, T_B_dev, QT_dev, means_A, means_B,
      norms_A, norms_B, df_A, df_B, dg_A, dg_B, scratchpad;
  unordered_map<int, DeviceProfile> profile_a_tile_dev, profile_b_tile_dev;
  unordered_map<int, cudaEvent_t> clocks_start, clocks_end, copy_to_host_done;
  unordered_map<int, cudaStream_t> streams;
  unordered_map<int, std::shared_ptr<fft_precompute_helper>> scratch;
  unordered_map<int, cudaDeviceProp> dev_props;
  SCAMPProfileType profile_type;
  size_t size_A;
  size_t size_B;
  size_t tile_size;
  size_t tile_n_x;
  size_t tile_n_y;
  size_t m;
  OptionalArgs opt_args;
  const bool self_join;
  const bool full_join;
  const size_t MAX_TILE_SIZE;
  const SCAMPPrecisionType fp_type;
  vector<int> devices;
  const size_t tile_start_row_position;
  const size_t tile_start_col_position;
  // Tile state variables
  list<pair<int, int>> tile_ordering;
  int completed_tiles;
  size_t total_tiles;
  unordered_map<int, size_t> n_x;
  unordered_map<int, size_t> n_y;
  unordered_map<int, size_t> n_x_2;
  unordered_map<int, size_t> n_y_2;
  unordered_map<int, size_t> pos_x;
  unordered_map<int, size_t> pos_y;
  unordered_map<int, size_t> pos_x_2;
  unordered_map<int, size_t> pos_y_2;

  SCAMPError_t do_tile(SCAMPTileType t, int device,
                       const google::protobuf::RepeatedField<double> &Ta_h,
                       const google::protobuf::RepeatedField<double> &Tb_h);

  bool pick_and_start_next_tile(
      int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  int issue_and_merge_tiles_on_devices(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b,
      Profile *profile_a_full, Profile *profile_b_full,
      vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
      int last_device_idx);
  int issue_and_merge_tiles_on_devices(const vector<double> &Ta_host,
                                       const vector<double> &Tb_host,
                                       vector<double> &profile_A_full_host,
                                       vector<double> &profile_B_full_host,
                                       vector<vector<double>> &profileA_h,
                                       vector<vector<double>> &profileB_h,
                                       int last_device_idx);
  void get_tile_ordering();
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length, cudaStream_t s);
  void MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                uint64_t length, Profile *full_profile);
  Profile InitProfile(SCAMPProfileType t, uint64_t size);
  SCAMPError_t InitInputOnDevice(
      const google::protobuf::RepeatedField<double> &Ta_h,
      const google::protobuf::RepeatedField<double> &Tb_h, int device);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, bool do_full_join, size_t start_row,
                  size_t start_col, OptionalArgs args_,
                  SCAMPProfileType profile_type_)
      : size_A(Asize),
        m(window_sz),
        MAX_TILE_SIZE(max_tile_size),
        devices(dev),
        self_join(selfjoin),
        completed_tiles(0),
        fp_type(t),
        full_join(do_full_join),
        tile_start_row_position(start_row),
        tile_start_col_position(start_col),
        opt_args(args_),
        profile_type(profile_type_) {
    if (self_join) {
      size_B = size_A;
    } else {
      size_B = Bsize;
    }
    tile_size = Asize / (devices.size());
    if (tile_size > MAX_TILE_SIZE) {
      tile_size = MAX_TILE_SIZE;
    }
    for (auto device : devices) {
      n_x.emplace(device, 0);
      n_y.emplace(device, 0);
      n_x_2.emplace(device, 0);
      n_y_2.emplace(device, 0);
      pos_x.emplace(device, 0);
      pos_y.emplace(device, 0);
      pos_x_2.emplace(device, 0);
      pos_y_2.emplace(device, 0);
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      dev_props.emplace(device, properties);
    }
    // n_y = Asize - m + 1;
    // n_x = Bsize - m + 1;
    tile_n_x = tile_size - m + 1;
    tile_n_y = tile_n_x;
  }
  SCAMPError_t do_join(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b,
      Profile *profile_a, Profile *profile_b);
  SCAMPError_t init();
  SCAMPError_t destroy();
};

}  // namespace SCAMP
