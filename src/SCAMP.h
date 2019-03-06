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
  unordered_map<int, double *> _T_A_dev, _T_B_dev, _QT_dev, _means_A, _means_B,
      _norms_A, _norms_B, _df_A, _df_B, _dg_A, _dg_B, _scratchpad;
  // TODO(zpzim): only use these for GPU computation, make a general
  // "DeviceProfile" for CPUs and GPUs
  unordered_map<int, DeviceProfile> _profile_a_tile_dev, _profile_b_tile_dev;
  // TODO(zpzim): do not rely on cudaEvents for timing and synchronization
  unordered_map<int, cudaEvent_t> _clocks_start, _clocks_end,
      _copy_to_host_done;
  // TODO(zpzim): SCAMP_Operation should not be required to use cuda streams,
  // this is preventing us from adding a CPU codepath to SCAMP
  unordered_map<int, cudaStream_t> _streams;
  unordered_map<int, std::shared_ptr<fft_precompute_helper>> _scratch;
  // TODO(zpzim): device properties should allow CPUs to be listed as devices as
  // well.
  unordered_map<int, cudaDeviceProp> _dev_props;
  // Result vectors
  Profile *_profile_a, *_profile_b;
  // Type of profile to compute
  SCAMPProfileType _profile_type;
  // Total size of A timeseries
  size_t _full_ts_len_A;
  // Total size of B timesereis
  size_t _full_ts_len_B;
  // Max size of the timeseries associated with the tile
  size_t _max_tile_ts_size;
  // Max width of the distance matrix associated with the tile
  size_t _max_tile_width;
  // Max height of the distance matrix associated with the tile
  size_t _max_tile_height;
  // Subsequence window length for MP
  size_t _mp_window;
  // Optional kernel arguments
  OptionalArgs _opt_args;
  // Whether or not we are computing a self join (symmetric distance matrix)
  const bool _self_join;
  // Whether or not to compute MP along the rows.
  const bool _computing_rows;
  // Whether or not to compute MP along the columns.
  const bool _computing_cols;
  // Whether or not time series A and B start with the same prefix.
  const bool _is_aligned;
  // Determines if we should keep the row/column matrix profiles separate or to
  // merge them.
  const bool _keep_rows_separate;
  // Absolute maximum length of a time series to use in a tile
  // TODO(zpzim): Make this the maximum length of the profile in a tile rather
  // than the time series.
  const size_t MAX_TILE_SIZE;
  // Precision type of computation
  const SCAMPPrecisionType _fp_type;
  // CUDA device ids to use for computation
  // TODO(zpzim): Convert this to a general device type (For CPU and GPU
  // computation)
  vector<int> _devices;

  // For distributed joins, the start position of this join in relation to other
  // distributed tiles.
  const int64_t _tile_start_row_position;
  const int64_t _tile_start_col_position;

  // Tile state variables
  // The order to compute the tiles in, set by get_tile_ordering()
  list<pair<int, int>> _tile_ordering;
  // The number of completed tiles
  int _completed_tiles;
  // The total number of tiles
  size_t _total_tiles;

  // Current and previous tile dimensions for each device
  // TODO(zpzim): refactor this into a tile struct and prepopulate dimensions of
  // every tile Don't rely on current and previous tile sizes, this will make it
  // easier to remove the tight coupling of cuda streams with SCAMP_Operation
  // methods.
  unordered_map<int, size_t> _current_tile_width;
  unordered_map<int, size_t> _current_tile_height;
  unordered_map<int, size_t> _previous_tile_width;
  unordered_map<int, size_t> _previous_tile_height;
  unordered_map<int, size_t> _current_tile_col;
  unordered_map<int, size_t> _current_tile_row;
  unordered_map<int, size_t> _previous_tile_col;
  unordered_map<int, size_t> _previous_tile_row;

  SCAMPError_t do_tile(SCAMPTileType t, int device,
                       const google::protobuf::RepeatedField<double> &Ta_h,
                       const google::protobuf::RepeatedField<double> &Tb_h);

  bool pick_and_start_next_tile(
      int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  int issue_and_merge_tiles_on_devices(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b,
      vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
      int last_device_idx);
  void get_tile_ordering();
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length, cudaStream_t s);
  void MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                uint64_t length, Profile *full_profile,
                                uint64_t index_start);
  Profile InitProfile(SCAMPProfileType t, uint64_t size);
  SCAMPError_t InitInputOnDevice(
      const google::protobuf::RepeatedField<double> &Ta_h,
      const google::protobuf::RepeatedField<double> &Tb_h, int device);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, bool do_full_join, int64_t start_row,
                  int64_t start_col, OptionalArgs args_,
                  SCAMPProfileType profile_type, Profile *pA, Profile *pB,
                  bool keep_rows, bool compute_rows, bool compute_cols,
                  bool is_aligned)
      : _full_ts_len_A(Asize),
        _mp_window(window_sz),
        MAX_TILE_SIZE(max_tile_size),
        _devices(dev),
        _self_join(selfjoin),
        _completed_tiles(0),
        _fp_type(t),
        _tile_start_row_position(start_row),
        _tile_start_col_position(start_col),
        _opt_args(args_),
        _profile_type(profile_type),
        _profile_a(pA),
        _profile_b(pB),
        _keep_rows_separate(keep_rows),
        _computing_rows(compute_rows),
        _computing_cols(compute_cols),
        _is_aligned(is_aligned) {
    if (_self_join) {
      _full_ts_len_B = _full_ts_len_A;
    } else {
      _full_ts_len_B = Bsize;
    }
    _max_tile_ts_size = Asize / (_devices.size());
    if (_max_tile_ts_size > MAX_TILE_SIZE) {
      _max_tile_ts_size = MAX_TILE_SIZE;
    }
    for (auto device : _devices) {
      _current_tile_width.emplace(device, 0);
      _current_tile_height.emplace(device, 0);
      _previous_tile_width.emplace(device, 0);
      _previous_tile_height.emplace(device, 0);
      _current_tile_col.emplace(device, 0);
      _current_tile_row.emplace(device, 0);
      _previous_tile_col.emplace(device, 0);
      _previous_tile_row.emplace(device, 0);
      // TODO(zpzim): remove dependency on cuda
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      _dev_props.emplace(device, properties);
    }
    _max_tile_width = _max_tile_ts_size - _mp_window + 1;
    _max_tile_height = _max_tile_width;
  }
  SCAMPError_t do_join(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  SCAMPError_t init();
  SCAMPError_t destroy();
};

}  // namespace SCAMP
