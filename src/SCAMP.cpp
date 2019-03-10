#include <cinttypes>
#include <future>
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
    cudaStreamDestroy(_streams.at(device));
  }
  return SCAMP_NO_ERROR;
}

void SCAMP_Operation::copy_statistics_for_tile(int device) {
  size_t bytes_a =
      (_current_tile_width[device] - _mp_window + 1) * sizeof(double);
  size_t bytes_b =
      (_current_tile_height[device] - _mp_window + 1) * sizeof(double);
  cudaMemcpyAsync(_norms_A[device],
                  _normsa_h.data() + _current_tile_col[device], bytes_a,
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_norms_B[device],
                  _normsb_h.data() + _current_tile_row[device], bytes_b,
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_df_A[device], _dfa_h.data() + _current_tile_col[device],
                  bytes_a, cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_df_B[device], _dfb_h.data() + _current_tile_row[device],
                  bytes_b, cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_dg_A[device], _dga_h.data() + _current_tile_col[device],
                  bytes_a, cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_dg_B[device], _dgb_h.data() + _current_tile_row[device],
                  bytes_b, cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_means_A[device],
                  _meansa_h.data() + _current_tile_col[device], bytes_a,
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpyAsync(_means_B[device],
                  _meansb_h.data() + _current_tile_row[device], bytes_b,
                  cudaMemcpyHostToDevice, _streams.at(device));
  gpuErrchk(cudaPeekAtLastError());
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

  copy_statistics_for_tile(device);

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
  InitInputOnDevice(Ta_h, Tb_h, device);

  SCAMP_Tile tile(t, _T_A_dev[device], _T_B_dev[device], _df_A[device],
                  _df_B[device], _dg_A[device], _dg_B[device], _norms_A[device],
                  _norms_B[device], _means_A[device], _means_B[device],
                  _QT_dev[device], &_profile_a_tile_dev[device],
                  &_profile_b_tile_dev[device], start_x, start_y,
                  _tile_start_col_position, _tile_start_row_position,
                  _is_aligned, _current_tile_height[device],
                  _current_tile_width[device], _mp_window, _scratch[device],
                  _dev_props.at(device), _fp_type, _profile_type, _opt_args);

  err = tile.execute(_streams.at(device));
  return err;
}

void SCAMP_Operation::get_tiles() {
  size_t num_tile_rows = ceil((_full_ts_len_B - _mp_window + 1) /
                              static_cast<double>(_max_tile_height));
  size_t num_tile_cols = ceil((_full_ts_len_A - _mp_window + 1) /
                              static_cast<double>(_max_tile_width));
  if (_self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        _work_queue.push(std::make_pair(diag,diag+offset));
        //.emplace_back(diag, diag + offset);
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      _work_queue.push(std::make_pair(i, num_tile_cols - 1));
      //_tile_ordering.emplace_back(i, num_tile_cols - 1);
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset, diag + offset));
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset + diag, offset));
      }
    }

    // Add the corner edge tile
    _work_queue.push(std::make_pair(num_tile_rows - 1, num_tile_cols - 1));

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      ++y;
    }
  }
  _total_tiles = _work_queue.size();
}

void SCAMP_Operation::MergeResult(int tid) {
  int device = _devices.at(tid);
  cudaSetDevice(device);
  gpuErrchk(cudaPeekAtLastError());
  // Set up a copy operation back to the host
  CopyProfileToHost(&_profile_a_tile[tid], &_profile_a_tile_dev[tid],
                      _current_tile_width[tid] - _mp_window + 1,
                      _streams[device]);
  if (_computing_rows) {
    CopyProfileToHost(&_profile_b_tile[tid], &_profile_b_tile_dev[tid],
                        _current_tile_height[tid] - _mp_window + 1,
                        _streams[device]);
  }
  // Wait for the previous work to be done
  cudaStreamSynchronize(_streams[device]);
  gpuErrchk(cudaPeekAtLastError());
  // Merge result
  MergeTileIntoFullProfile(&_profile_a_tile[tid], _current_tile_col[tid],
                           _current_tile_width[tid] - _mp_window + 1,
                           _profile_a, _current_tile_row[tid], _profile_a_lock);
  if (_self_join) {
    MergeTileIntoFullProfile(&_profile_b_tile[tid],
                             _current_tile_row[tid],
                             _current_tile_height[tid] - _mp_window + 1,
                             _profile_a, _current_tile_col[tid], _profile_a_lock);
  } else if (_computing_rows && _keep_rows_separate) {
    MergeTileIntoFullProfile(&_profile_b_tile[tid],
                             _current_tile_row[tid],
                             _current_tile_height[tid] - _mp_window + 1,
                             _profile_b, _current_tile_col[tid], _profile_b_lock);
  }
}

void SCAMP_Operation::do_work(
    int tid, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  
  // TODO(zpzim): make CPU/GPU agnostic
  int device = _devices.at(tid);
  cudaSetDevice(device);
  
  while (!_work_queue.empty()) {
    std::pair<int,int> tile = _work_queue.pop();
    if (tile.first == -1 && tile.second == -1) {
        // Another thread grabbed our tile and now the queue is empty
        return;
    }
    // Get the position of the tile we will compute
    _current_tile_row[tid] = tile.first * _max_tile_height;
    _current_tile_col[tid] = tile.second * _max_tile_width;
    // Get the size of the tile we will compute
    _current_tile_width[tid] =
      std::min(_max_tile_ts_size, _full_ts_len_A - _current_tile_col[tid]);
    _current_tile_height[tid] =
      std::min(_max_tile_ts_size, _full_ts_len_B - _current_tile_row[tid]);
    std::cout << "Starting tile with starting row of " << _current_tile_row[tid]
            << " starting column of " << _current_tile_col[tid]
            << " with height " << _current_tile_height[tid] << " and width "
            << _current_tile_width[tid] << std::endl;
    SCAMPError_t err;
    if (_self_join) {
      if (tile.first == tile.second) {
          // partial tile on diagonal
          err =
              do_tile(SELF_JOIN_UPPER_TRIANGULAR, tid, timeseries_a, timeseries_b);
        } else {
          // full tile
          err = do_tile(SELF_JOIN_FULL_TILE, tid, timeseries_a, timeseries_b);
        }
    } else if (_computing_rows) {
        // BiDirectional AB-join
        err = do_tile(AB_FULL_JOIN_FULL_TILE, tid, timeseries_a, timeseries_b);
    } else {
        // Column AB-join
        err = do_tile(AB_JOIN_FULL_TILE, tid, timeseries_a, timeseries_b);
    }
    if (err != SCAMP_NO_ERROR) {
      printf("ERROR %d executing tile. \n", err);
    }
    // Merge join result
    MergeResult(tid);
    // FIXME: Protect with LOCK
    _completed_tiles++;
    
  
  }
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
                                               uint64_t index_start, std::mutex &lock) {
  std::unique_lock<std::mutex> mlock(lock);
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

SCAMPError_t SCAMP_Operation::do_join(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {

  
  const int num_workers = _devices.size();

  // Generate temporary result storage for each worker
  _profile_a_tile = std::vector<Profile>(num_workers, InitProfile(_profile_type, _max_tile_height));
  _profile_b_tile = std::vector<Profile>(num_workers, InitProfile(_profile_type, _max_tile_width));
  
  // Compute statistics
  compute_statistics(timeseries_a, &_normsa_h, &_dfa_h, &_dga_h, &_meansa_h,
                     _mp_window);
  compute_statistics(timeseries_b, &_normsb_h, &_dfb_h, &_dgb_h, &_meansb_h,
                     _mp_window);

  // Populate Work Queue with tiles
  get_tiles();

  std::cout << "Performing join with " << _work_queue.size() << " tiles."
            << std::endl;
  std::vector<std::future<void>> futures(num_workers);

  // Start workers
  for (int i = 0; i < num_workers; ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this, i, timeseries_a, timeseries_b);
  }

  // wait for workers to be done
  for (auto& future : futures) {
    future.get();
  }
  
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
