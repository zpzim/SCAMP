#include <cinttypes>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "SCAMP.h"
#include "common.h"
#include "tile.h"
#include "utils.h"
using std::vector;

namespace SCAMP {

static const int ISSUED_ALL_DEVICES = -2;

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_full[i + merge_start] += to_merge[i];
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge, uint64_t index_offset) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_entry e1, e2;
    e1.ulong = mp_full[i + merge_start];
    e2.ulong = to_merge[i];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[i + merge_start] = e2.ulong;
    }
  }
}

SCAMPError_t SCAMP_Operation::init() {
  for (auto device : _devices) {
    // TODO(zpzim): Remove dependancy on CUDA, have CPU and GPU specific
    // codepaths
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());

    _T_A_dev.insert({device, nullptr});
    _T_B_dev.insert({device, nullptr});
    _QT_dev.insert({device, nullptr});
    _means_A.insert({device, nullptr});
    _means_B.insert({device, nullptr});
    _norms_A.insert({device, nullptr});
    _norms_B.insert({device, nullptr});
    _df_A.insert({device, nullptr});
    _df_B.insert({device, nullptr});
    _dg_A.insert({device, nullptr});
    _dg_B.insert({device, nullptr});
    DeviceProfile d;
    d[_profile_type] = nullptr;
    _profile_a_tile_dev.insert({device, d});
    _profile_b_tile_dev.insert({device, d});
    _scratchpad.insert({device, nullptr});

    size_t profile_size = GetProfileTypeSize(_profile_type);
    // TODO(zpzim): Move these to a CUDA specific codepath, else use malloc/new
    cudaMalloc(&_T_A_dev.at(device), sizeof(double) * _max_tile_ts_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_T_B_dev.at(device), sizeof(double) * _max_tile_ts_size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_profile_a_tile_dev.at(device).at(_profile_type),
               profile_size * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_profile_b_tile_dev.at(device).at(_profile_type),
               profile_size * _max_tile_height);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_QT_dev.at(device), sizeof(double) * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_means_A.at(device), sizeof(double) * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_means_B.at(device), sizeof(double) * _max_tile_height);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_norms_A.at(device), sizeof(double) * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_norms_B.at(device), sizeof(double) * _max_tile_height);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_df_A.at(device), sizeof(double) * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_df_B.at(device), sizeof(double) * _max_tile_height);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_dg_A.at(device), sizeof(double) * _max_tile_width);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_dg_B.at(device), sizeof(double) * _max_tile_height);
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc(&_scratchpad.at(device), sizeof(double) * _max_tile_ts_size);
    _scratch[device] = std::make_shared<fft_precompute_helper>(
        _max_tile_ts_size, _mp_window, true);
    // TODO(zpzim): make CPU/GPU agnostic
    cudaEvent_t st, ed, copy;
    cudaEventCreate(&ed);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventCreate(&st);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventCreate(&copy);
    gpuErrchk(cudaPeekAtLastError());

    // TODO(zpzim): make CPU/GPU agnostic
    _clocks_start.emplace(device, st);
    _clocks_end.emplace(device, ed);
    _copy_to_host_done.emplace(device, copy);
    cudaStream_t s;
    cudaStreamCreate(&s);
    gpuErrchk(cudaPeekAtLastError());
    _streams.emplace(device, s);
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::destroy() {
  for (auto device : _devices) {
    // TODO(zpzim): make CPU/GPU agnostic
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(_T_A_dev[device]);
    cudaFree(_T_B_dev[device]);
    cudaFree(_QT_dev[device]);
    cudaFree(_means_A[device]);
    cudaFree(_means_B[device]);
    cudaFree(_norms_A[device]);
    cudaFree(_norms_B[device]);
    cudaFree(_df_A[device]);
    cudaFree(_df_B[device]);
    cudaFree(_dg_A[device]);
    cudaFree(_dg_B[device]);
    cudaFree(_profile_a_tile_dev[device].at(_profile_type));
    cudaFree(_profile_b_tile_dev[device].at(_profile_type));
    cudaFree(_scratchpad.at(device));
    cudaEventDestroy(_clocks_start[device]);
    cudaEventDestroy(_clocks_end[device]);
    cudaEventDestroy(_copy_to_host_done[device]);
    cudaStreamDestroy(_streams.at(device));
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::InitInputOnDevice(
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h, int device) {
  int profile_size = GetProfileTypeSize(_profile_type);

  // TODO(zpzim): make CPU/GPU agnostic
  cudaMemcpyAsync(_T_A_dev[device], Ta_h.data() + _current_tile_col[device],
                  sizeof(double) * _current_tile_width[device],
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_T_B_dev[device], Tb_h.data() + _current_tile_row[device],
                  sizeof(double) * _current_tile_height[device],
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());

  // TODO(zpzim): make CPU/GPU agnostic
  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      cudaMemsetAsync(
          _profile_a_tile_dev.at(device).at(_profile_type), 0,
          profile_size * (_current_tile_width[device] - _mp_window + 1),
          _streams.at(device));
      gpuErrchk(cudaPeekAtLastError());
      cudaMemsetAsync(
          _profile_b_tile_dev.at(device).at(_profile_type), 0,
          profile_size * (_current_tile_height[device] - _mp_window + 1),
          _streams.at(device));
      gpuErrchk(cudaPeekAtLastError());
      break;
    case PROFILE_TYPE_1NN_INDEX: {
      const uint64_t *pA_ptr =
          _profile_a->data().Get(0).uint64_value().value().data();
      cudaMemcpyAsync(
          _profile_a_tile_dev.at(device).at(_profile_type),
          pA_ptr + _current_tile_col[device],
          sizeof(uint64_t) * (_current_tile_width[device] - _mp_window + 1),
          cudaMemcpyHostToDevice, _streams.at(device));
      gpuErrchk(cudaPeekAtLastError());
      if (_self_join) {
        cudaMemcpyAsync(
            _profile_b_tile_dev.at(device).at(_profile_type),
            pA_ptr + _current_tile_row[device],
            sizeof(uint64_t) * (_current_tile_height[device] - _mp_window + 1),
            cudaMemcpyHostToDevice, _streams.at(device));
        gpuErrchk(cudaPeekAtLastError());

      } else if (_computing_rows && _keep_rows_separate) {
        const uint64_t *pB_ptr =
            _profile_b->data().Get(0).uint64_value().value().data();
        cudaMemcpyAsync(
            _profile_b_tile_dev.at(device).at(_profile_type),
            pB_ptr + _current_tile_row[device],
            sizeof(uint64_t) * (_current_tile_height[device] - _mp_window + 1),
            cudaMemcpyHostToDevice, _streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
      }
      break;
    }
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    case PROFILE_TYPE_INVALID:
      return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::do_tile(
    SCAMPTileType t, int device,
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h) {
  size_t start_x = _current_tile_col[device];
  size_t start_y = _current_tile_row[device];
  SCAMPError_t err;
  size_t t_n_x = _current_tile_width[device] - _mp_window + 1;
  size_t t_n_y = _current_tile_height[device] - _mp_window + 1;
  InitInputOnDevice(Ta_h, Tb_h, device);

  // FIXME: Computing the sliding dot products & statistics for each tile is
  // overkill, we should precompute for the whole SCAMP_Operation
  compute_statistics(_T_A_dev[device], _norms_A[device], _df_A[device],
                     _dg_A[device], _means_A[device], t_n_x, _mp_window,
                     _streams.at(device), _scratchpad[device]);
  gpuErrchk(cudaPeekAtLastError());
  compute_statistics(_T_B_dev[device], _norms_B[device], _df_B[device],
                     _dg_B[device], _means_B[device], t_n_y, _mp_window,
                     _streams.at(device), _scratchpad[device]);
  gpuErrchk(cudaPeekAtLastError());
  SCAMP_Tile tile(t, _T_A_dev[device], _T_B_dev[device], _df_A[device],
                  _df_B[device], _dg_A[device], _dg_B[device], _norms_A[device],
                  _norms_B[device], _means_A[device], _means_B[device],
                  _QT_dev[device], &_profile_a_tile_dev[device],
                  &_profile_b_tile_dev[device], start_x, start_y,
                  _tile_start_col_position, _tile_start_row_position,
                  _is_aligned, _current_tile_height[device],
                  _current_tile_width[device], _mp_window, _scratch[device],
                  _dev_props.at(device), _fp_type, _profile_type, _opt_args);

  // TODO(zpzim): make CPU/GPU agnostic
  cudaEventRecord(_clocks_start[device], _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  err = tile.execute(_streams.at(device));
  cudaEventRecord(_clocks_end[device], _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  return err;
}

void SCAMP_Operation::get_tile_ordering() {
  _tile_ordering.clear();
  size_t num_tile_rows = ceil((_full_ts_len_B - _mp_window + 1) /
                              static_cast<double>(_max_tile_height));
  size_t num_tile_cols = ceil((_full_ts_len_A - _mp_window + 1) /
                              static_cast<double>(_max_tile_width));
  if (_self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        _tile_ordering.emplace_back(diag, diag + offset);
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      _tile_ordering.emplace_back(i, num_tile_cols - 1);
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        _tile_ordering.emplace_back(offset, diag + offset);
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        _tile_ordering.emplace_back(offset + diag, offset);
      }
    }

    // Add the corner edge tile
    _tile_ordering.emplace_back(num_tile_rows - 1, num_tile_cols - 1);

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      _tile_ordering.emplace_back(y, num_tile_cols - 1);
      _tile_ordering.emplace_back(num_tile_rows - 1, x);
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      _tile_ordering.emplace_back(num_tile_rows - 1, x);
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      _tile_ordering.emplace_back(y, num_tile_cols - 1);
      ++y;
    }
  }
  _total_tiles = _tile_ordering.size();
}

bool SCAMP_Operation::pick_and_start_next_tile(
    int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  bool done = false;
  int tile_row = _tile_ordering.front().first;
  int tile_col = _tile_ordering.front().second;
  // Get the position of the tile we will compute
  _current_tile_col[dev] = tile_col * _max_tile_width;
  _current_tile_row[dev] = tile_row * _max_tile_height;
  // Get the size of the tile we will compute
  _current_tile_width[dev] =
      std::min(_max_tile_ts_size, _full_ts_len_A - _current_tile_col[dev]);
  _current_tile_height[dev] =
      std::min(_max_tile_ts_size, _full_ts_len_B - _current_tile_row[dev]);
  std::cout << "Starting tile with starting row of " << _current_tile_row[dev]
            << " starting column of " << _current_tile_row[dev]
            << " with height " << _current_tile_height[dev] << " and width "
            << _current_tile_width[dev] << std::endl;
  SCAMPError_t err;
  if (_self_join) {
    if (tile_row == tile_col) {
      // partial tile on diagonal
      err =
          do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, timeseries_a, timeseries_b);
    } else {
      // full tile
      err = do_tile(SELF_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
    }
  } else if (_computing_rows) {
    // BiDirectional AB-join
    err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
  } else {
    // Column AB-join
    err = do_tile(AB_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b);
  }
  if (err != SCAMP_NO_ERROR) {
    printf("ERROR %d executing tile. \n", err);
  }
  _tile_ordering.pop_front();
  if (_tile_ordering.empty()) {
    done = true;
  }
  return done;
}

// TODO(zpzim): make CPU/GPU agnostic
void SCAMP_Operation::CopyProfileToHost(
    Profile *destination_profile, const DeviceProfile *device_tile_profile,
    uint64_t length, cudaStream_t s) {
  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      cudaMemcpyAsync(destination_profile->mutable_data()
                          ->Mutable(0)
                          ->mutable_double_value()
                          ->mutable_value()
                          ->mutable_data(),
                      device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
                      length * sizeof(double), cudaMemcpyDeviceToHost, s);
      gpuErrchk(cudaPeekAtLastError());
      break;
    case PROFILE_TYPE_1NN_INDEX:
      cudaMemcpyAsync(destination_profile->mutable_data()
                          ->Mutable(0)
                          ->mutable_uint64_value()
                          ->mutable_value()
                          ->mutable_data(),
                      device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
                      length * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
      gpuErrchk(cudaPeekAtLastError());
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      break;
  }
}

void SCAMP_Operation::MergeTileIntoFullProfile(Profile *tile_profile,
                                               uint64_t position,
                                               uint64_t length,
                                               Profile *full_profile,
                                               uint64_t index_start = 0) {
  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data(),
                              position, length,
                              tile_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(full_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                index_start);
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<uint64_t>(full_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data());
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return;
  }
}

// TODO(zpzim): make CPU/GPU agnostic
// TODO(zpzim): change to a work queue rather than round-robin scheduling
int SCAMP_Operation::issue_and_merge_tiles_on_devices(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b,
    vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
    int last_device_idx = ISSUED_ALL_DEVICES) {
  bool done = last_device_idx != ISSUED_ALL_DEVICES;
  int last_dev = ISSUED_ALL_DEVICES;
  if (last_device_idx == ISSUED_ALL_DEVICES) {
    last_device_idx = _devices.size() - 1;
  }
  // Grab the completed tiles from the device
  for (int i = 0; i <= last_device_idx; ++i) {
    int device = _devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    // Set up a copy operation back to the host
    CopyProfileToHost(&profile_a_tile->at(i), &_profile_a_tile_dev[device],
                      _current_tile_width[device] - _mp_window + 1,
                      _streams[device]);
    if (_computing_rows) {
      CopyProfileToHost(&profile_b_tile->at(i), &_profile_b_tile_dev[device],
                        _current_tile_height[device] - _mp_window + 1,
                        _streams[device]);
    }
    cudaEventRecord(_copy_to_host_done[device], _streams.at(device));
    gpuErrchk(cudaPeekAtLastError());
    // Save the current tile dimensions so we can copy the result back later
    _previous_tile_width[device] = _current_tile_width[device];
    _previous_tile_height[device] = _current_tile_height[device];
    _previous_tile_col[device] = _current_tile_col[device];
    _previous_tile_row[device] = _current_tile_row[device];
    // Start the next tile
    if (!done) {
      done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
      if (done) {
        last_dev = i;
      }
    }
  }

  for (int i = 0; i <= last_device_idx; ++i) {
    int device = _devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    // Wait for the previous work to be done
    cudaEventSynchronize(_copy_to_host_done[device]);
    gpuErrchk(cudaPeekAtLastError());
    // Merge result
    MergeTileIntoFullProfile(&profile_a_tile->at(i), _previous_tile_col[device],
                             _previous_tile_width[device] - _mp_window + 1,
                             _profile_a, _previous_tile_row[device]);
    if (_self_join) {
      MergeTileIntoFullProfile(&profile_b_tile->at(i),
                               _previous_tile_row[device],
                               _previous_tile_height[device] - _mp_window + 1,
                               _profile_a, _previous_tile_col[device]);

    } else if (_computing_rows && _keep_rows_separate) {
      MergeTileIntoFullProfile(&profile_b_tile->at(i),
                               _previous_tile_row[device],
                               _previous_tile_height[device] - _mp_window + 1,
                               _profile_b, _previous_tile_col[device]);
    }
    _completed_tiles++;
  }
  std::cout << _completed_tiles / static_cast<float>(_total_tiles) * 100
            << " percent complete." << std::endl;
  return last_dev;
}

Profile SCAMP_Operation::InitProfile(SCAMPProfileType t, uint64_t size) {
  Profile p;
  p.set_type(t);
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      p.mutable_data()->Add()->mutable_double_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = std::numeric_limits<float>::lowest();
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, e.ulong);
      return p;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return p;
  }
}

// TODO(zpzim): make CPU/GPU agnostic
SCAMPError_t SCAMP_Operation::do_join(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  vector<Profile> profile_a_tile(_devices.size(),
                                 InitProfile(_profile_type, _max_tile_height));
  vector<Profile> profile_b_tile(_devices.size(),
                                 InitProfile(_profile_type, _max_tile_width));

  bool done = false;
  int last_dev = ISSUED_ALL_DEVICES;
  get_tile_ordering();
  std::cout << "Performing join with " << _tile_ordering.size() << " tiles."
            << std::endl;
  // Start the first tile on each device
  for (int i = 0; i < _devices.size(); ++i) {
    int device = _devices.at(i);
    cudaSetDevice(device);
    gpuErrchk(cudaPeekAtLastError());
    done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
    gpuErrchk(cudaPeekAtLastError());
    if (done) {
      last_dev = i;
      break;
    }
  }

  while (last_dev == ISSUED_ALL_DEVICES) {
    // Finish the current tile on each device and start the next one
    last_dev = issue_and_merge_tiles_on_devices(
        timeseries_a, timeseries_b, &profile_a_tile, &profile_b_tile);
  }
  // Finish the last tile on each device
  issue_and_merge_tiles_on_devices(timeseries_a, timeseries_b, &profile_a_tile,
                                   &profile_b_tile, last_dev);

  return SCAMP_NO_ERROR;
}

// TODO(zpzim): Make CPU/GPU agnostic
void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices) {
  if (devices.empty()) {
    printf("Error: no gpu provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  OptionalArgs _opt_args(args->distance_threshold());
  // Construct operation
  SCAMP_Operation op(
      args->timeseries_a().size(), args->timeseries_b().size(), args->window(),
      args->max_tile_size(), devices, !args->has_b(), args->precision_type(),
      args->computing_columns() && args->computing_rows(),
      args->distributed_start_row(), args->distributed_start_col(), _opt_args,
      args->profile_type(), args->mutable_profile_a(),
      args->mutable_profile_b(), args->keep_rows_separate(),
      args->computing_rows(), args->computing_columns(), args->is_aligned());
  // Init memory
  op.init();
  gpuErrchk(cudaPeekAtLastError());
  start = clock();
  // Execute op
  if (args->has_b()) {
    op.do_join(args->timeseries_a(), args->timeseries_b());
  } else {
    op.do_join(args->timeseries_a(), args->timeseries_a());
  }
  cudaDeviceSynchronize();
  end = clock();
  gpuErrchk(cudaPeekAtLastError());
  op.destroy();
  gpuErrchk(cudaPeekAtLastError());
  printf(
      "Finished SCAMP to generate  matrix profile in %f "
      "seconds on %lu devices:\n",
      (end - start) / static_cast<double>(CLOCKS_PER_SEC), devices.size());
}

}  // namespace SCAMP
