#include "tile.h"
#ifdef _HAS_CUDA_
#include "kernels.h"
#endif
#include "cpu_kernels.h"

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

Tile::Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id)
    : _info(info),
      _arch(arch),
      _cuda_id(cuda_id),
      _current_tile_width(0),
      _current_tile_height(0),
      _current_tile_row(0),
      _current_tile_col(0) {
  _profile_a_tile_dev[_info->profile_type] = nullptr;
  _profile_b_tile_dev[_info->profile_type] = nullptr;
  _profile_a_tile = AllocProfile(_info->profile_type, _info->max_tile_height);
  _profile_b_tile = AllocProfile(_info->profile_type, _info->max_tile_width);
  switch (_arch) {
    case CUDA_GPU_WORKER:
      init_cuda();
      break;
    case CPU_WORKER:
      init_cpu();
      break;
  }
}

Tile::~Tile() {
  switch (_arch) {
    case CUDA_GPU_WORKER:
      free_cuda();
      break;
    case CPU_WORKER:
      free_cpu();
      break;
  }
}

void Tile::Sync() {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      cudaStreamSynchronize(_stream);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      break;
  }
}

void Tile::InitTimeseries(const google::protobuf::RepeatedField<double> &Ta_h,
                          const google::protobuf::RepeatedField<double> &Tb_h) {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      cudaMemcpyAsync(_T_A_dev, Ta_h.data() + _current_tile_col,
                      sizeof(double) * _current_tile_width,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_T_B_dev, Tb_h.data() + _current_tile_row,
                      sizeof(double) * _current_tile_height,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      // TODO(zpzim): we don't actually have to copy memory here, we
      // can just set a reference.
      memcpy(_T_A_dev, Ta_h.data() + _current_tile_col,
             sizeof(double) * _current_tile_width);
      memcpy(_T_B_dev, Tb_h.data() + _current_tile_row,
             sizeof(double) * _current_tile_height);
      break;
  }
}

Profile Tile::AllocProfile(SCAMPProfileType t, uint64_t size) {
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

SCAMPError_t Tile::InitProfile(Profile *profile_a, Profile *profile_b) {
  int profile_size = GetProfileTypeSize(_info->profile_type);
  int width = _current_tile_width - _info->mp_window + 1;
  int height = _current_tile_height - _info->mp_window + 1;
  SCAMPProfileType type = _info->profile_type;
  switch (_arch) {
    case CUDA_GPU_WORKER: {
#if _HAS_CUDA_
      switch (type) {
        case PROFILE_TYPE_SUM_THRESH:
          cudaMemsetAsync(_profile_a_tile_dev.at(type), 0, profile_size * width,
                          _stream);
          gpuErrchk(cudaPeekAtLastError());
          cudaMemsetAsync(_profile_b_tile_dev.at(type), 0,
                          profile_size * height, _stream);
          gpuErrchk(cudaPeekAtLastError());
          break;
        case PROFILE_TYPE_1NN_INDEX: {
          const uint64_t *pA_ptr =
              profile_a->data().Get(0).uint64_value().value().data();
          cudaMemcpyAsync(_profile_a_tile_dev.at(type),
                          pA_ptr + _current_tile_col, sizeof(uint64_t) * width,
                          cudaMemcpyHostToDevice, _stream);
          gpuErrchk(cudaPeekAtLastError());
          if (_info->self_join) {
            cudaMemcpyAsync(
                _profile_b_tile_dev.at(type), pA_ptr + _current_tile_row,
                sizeof(uint64_t) * height, cudaMemcpyHostToDevice, _stream);
            gpuErrchk(cudaPeekAtLastError());

          } else if (_info->computing_rows && _info->keep_rows_separate) {
            const uint64_t *pB_ptr =
                profile_b->data().Get(0).uint64_value().value().data();
            cudaMemcpyAsync(
                _profile_b_tile_dev.at(type), pB_ptr + _current_tile_row,
                sizeof(uint64_t) * height, cudaMemcpyHostToDevice, _stream);
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
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    }
    // TODO(zpzim): Implement CPU codepath
    case CPU_WORKER:
      switch (_info->profile_type) {
        case PROFILE_TYPE_1NN_INDEX: {
          const uint64_t *pA_ptr =
              profile_a->data().Get(0).uint64_value().value().data();
          memcpy(_profile_a_tile_dev.at(type), pA_ptr + _current_tile_col,
                 sizeof(uint64_t) * width);
          if (_info->self_join) {
            memcpy(_profile_b_tile_dev.at(type), pA_ptr + _current_tile_row,
                   sizeof(uint64_t) * height);
          } else if (_info->computing_rows && _info->keep_rows_separate) {
            const uint64_t *pB_ptr =
                profile_b->data().Get(0).uint64_value().value().data();
            memcpy(_profile_b_tile_dev.at(type), pB_ptr + _current_tile_row,
                   sizeof(uint64_t) * height);
          }
          break;
        }

        case PROFILE_TYPE_SUM_THRESH:
        case PROFILE_TYPE_FREQUENCY_THRESH:
        case PROFILE_TYPE_KNN:
        case PROFILE_TYPE_1NN_MULTIDIM:
        case PROFILE_TYPE_INVALID:
          return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
      }
      break;
  }
  return SCAMP_NO_ERROR;
}

void Tile::InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b) {
  size_t bytes_a =
      (_current_tile_width - _info->mp_window + 1) * sizeof(double);
  size_t bytes_b =
      (_current_tile_height - _info->mp_window + 1) * sizeof(double);
  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaMemcpyAsync(_norms_A, a.norms().data() + _current_tile_col, bytes_a,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_df_A, a.df().data() + _current_tile_col, bytes_a,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_dg_A, a.dg().data() + _current_tile_col, bytes_a,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_means_A, a.means().data() + _current_tile_col, bytes_a,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_norms_B, b.norms().data() + _current_tile_row, bytes_b,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_df_B, b.df().data() + _current_tile_row, bytes_b,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_dg_B, b.dg().data() + _current_tile_row, bytes_b,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_means_B, b.means().data() + _current_tile_row, bytes_b,
                      cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      // TODO(zpzim): we don't actually have to copy memory here, we
      // can just set a reference.
      memcpy(_norms_A, a.norms().data() + _current_tile_col, bytes_a);
      memcpy(_df_A, a.df().data() + _current_tile_col, bytes_a);
      memcpy(_dg_A, a.dg().data() + _current_tile_col, bytes_a);
      memcpy(_means_A, a.means().data() + _current_tile_col, bytes_a);
      memcpy(_norms_B, b.norms().data() + _current_tile_row, bytes_b);
      memcpy(_df_B, b.df().data() + _current_tile_row, bytes_b);
      memcpy(_dg_B, b.dg().data() + _current_tile_row, bytes_b);
      memcpy(_means_B, b.means().data() + _current_tile_row, bytes_b);
      break;
  }
}

void Tile::MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                    uint64_t length, Profile *full_profile,
                                    uint64_t index_start, std::mutex &lock) {
  std::unique_lock<std::mutex> mlock(lock);
  switch (_info->profile_type) {
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

void Tile::MergeProfile(Profile *profile_a, std::mutex &a_lock,
                        Profile *profile_b, std::mutex &b_lock) {
  // Set up a copy operation back to the host
  CopyProfileToHost(&_profile_a_tile, &_profile_a_tile_dev,
                    _current_tile_width - _info->mp_window + 1);
  if (_info->computing_rows) {
    CopyProfileToHost(&_profile_b_tile, &_profile_b_tile_dev,
                      _current_tile_height - _info->mp_window + 1);
  }

  // Wait for the previous work to be done
  Sync();

  // Merge result
  MergeTileIntoFullProfile(&_profile_a_tile, _current_tile_col,
                           _current_tile_width - _info->mp_window + 1,
                           profile_a, _current_tile_row, a_lock);
  if (_info->self_join) {
    MergeTileIntoFullProfile(&_profile_b_tile, _current_tile_row,
                             _current_tile_height - _info->mp_window + 1,
                             profile_a, _current_tile_col, a_lock);
  } else if (_info->computing_rows && _info->keep_rows_separate) {
    MergeTileIntoFullProfile(&_profile_b_tile, _current_tile_row,
                             _current_tile_height - _info->mp_window + 1,
                             profile_b, _current_tile_col, b_lock);
  }
}
// TODO(zpzim): make CPU/GPU agnostic
void Tile::CopyProfileToHost(Profile *destination_profile,
                             const DeviceProfile *device_tile_profile,
                             uint64_t length) {
  switch (_arch) {
    case CUDA_GPU_WORKER: {
#ifdef _HAS_CUDA_
      switch (_info->profile_type) {
        case PROFILE_TYPE_SUM_THRESH:
          cudaMemcpyAsync(destination_profile->mutable_data()
                              ->Mutable(0)
                              ->mutable_double_value()
                              ->mutable_value()
                              ->mutable_data(),
                          device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
                          length * sizeof(double), cudaMemcpyDeviceToHost,
                          _stream);
          gpuErrchk(cudaPeekAtLastError());
          break;
        case PROFILE_TYPE_1NN_INDEX:
          cudaMemcpyAsync(destination_profile->mutable_data()
                              ->Mutable(0)
                              ->mutable_uint64_value()
                              ->mutable_value()
                              ->mutable_data(),
                          device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
                          length * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                          _stream);
          gpuErrchk(cudaPeekAtLastError());
          break;
        case PROFILE_TYPE_FREQUENCY_THRESH:
        case PROFILE_TYPE_KNN:
        case PROFILE_TYPE_1NN_MULTIDIM:
        default:
          break;
      }
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    }
    case CPU_WORKER:
      // TODO(zpzim): implement stub
      break;
  }
}

inline void Tile::init_cuda() {
#ifdef _HAS_CUDA_
  cudaSetDevice(_cuda_id);
  cudaGetDeviceProperties(&_dev_props, _cuda_id);
  cudaStreamCreate(&_stream);
  size_t profile_size = GetProfileTypeSize(_info->profile_type);
  cudaMalloc(&_T_A_dev, sizeof(double) * _info->max_tile_ts_size);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_T_B_dev, sizeof(double) * _info->max_tile_ts_size);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_profile_a_tile_dev.at(_info->profile_type),
             profile_size * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_profile_b_tile_dev.at(_info->profile_type),
             profile_size * _info->max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_QT_dev, sizeof(double) * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_means_A, sizeof(double) * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_means_B, sizeof(double) * _info->max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_norms_A, sizeof(double) * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_norms_B, sizeof(double) * _info->max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_df_A, sizeof(double) * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_df_B, sizeof(double) * _info->max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_dg_A, sizeof(double) * _info->max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_dg_B, sizeof(double) * _info->max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_scratchpad, sizeof(double) * _info->max_tile_ts_size);
  _scratch = std::make_shared<fft_precompute_helper>(
      _info->max_tile_ts_size, _info->mp_window, true, CUDA_GPU_WORKER);
#else
  assert("ERROR: CUDA used in binary not built with CUDA");
#endif
}

inline void Tile::free_cuda() {
#ifdef _HAS_CUDA_
  cudaSetDevice(_cuda_id);
  cudaFree(_T_A_dev);
  cudaFree(_T_B_dev);
  cudaFree(_QT_dev);
  cudaFree(_means_A);
  cudaFree(_means_B);
  cudaFree(_norms_A);
  cudaFree(_norms_B);
  cudaFree(_df_A);
  cudaFree(_df_B);
  cudaFree(_dg_A);
  cudaFree(_dg_B);
  cudaFree(_profile_a_tile_dev.at(_info->profile_type));
  cudaFree(_profile_b_tile_dev.at(_info->profile_type));
  cudaFree(_scratchpad);
  cudaStreamDestroy(_stream);
#else
  assert("ERROR: CUDA used in binary not built with CUDA");
#endif
}

// TODO(zpzim): Finish STUB
inline void Tile::init_cpu() {
  size_t profile_size = GetProfileTypeSize(_info->profile_type);
  _T_A_dev =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_ts_size));
  _T_B_dev =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_ts_size));
  _profile_a_tile_dev.at(_info->profile_type) =
      static_cast<double *>(malloc(profile_size * _info->max_tile_width));
  _profile_b_tile_dev.at(_info->profile_type) =
      static_cast<double *>(malloc(profile_size * _info->max_tile_height));
  _QT_dev =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_width));
  _means_A =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_width));
  _means_B =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_height));
  _norms_A =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_width));
  _norms_B =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_height));
  _df_A = static_cast<double *>(malloc(sizeof(double) * _info->max_tile_width));
  _df_B =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_height));
  _dg_A = static_cast<double *>(malloc(sizeof(double) * _info->max_tile_width));
  _dg_B =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_height));
  _scratchpad =
      static_cast<double *>(malloc(sizeof(double) * _info->max_tile_ts_size));
  _scratch = std::make_shared<fft_precompute_helper>(
      _info->max_tile_ts_size, _info->mp_window, true, CPU_WORKER);
}

// TODO(zpzim): Finish STUB
inline void Tile::free_cpu() {
  free(_T_A_dev);
  free(_T_B_dev);
  free(_QT_dev);
  free(_means_A);
  free(_means_B);
  free(_norms_A);
  free(_norms_B);
  free(_df_A);
  free(_df_B);
  free(_dg_A);
  free(_dg_B);
  free(_profile_a_tile_dev.at(_info->profile_type));
  free(_profile_b_tile_dev.at(_info->profile_type));
  free(_scratchpad);
}

SCAMPError_t Tile::execute(SCAMPTileType t) {
  SCAMPError_t error;
  switch (t) {
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
      error = do_ab_join_full();
      break;
    default:
      error = SCAMP_TILE_ILLEGAL_TYPE;
      break;
  }
  return error;
}

SCAMPError_t Tile::do_self_join_full() {
  SCAMPError_t error = SCAMP_NO_ERROR;

  error = do_self_join_half();
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          _scratch->compute_QT(_QT_dev, _T_B_dev, _T_A_dev, _means_A, _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_lower(
          _QT_dev, _df_A, _df_B, _dg_A, _dg_B, _norms_A, _norms_B,
          &_profile_a_tile_dev, &_profile_b_tile_dev, _info->mp_window,
          _current_tile_width - _info->mp_window + 1,
          _current_tile_height - _info->mp_window + 1, _current_tile_col,
          _current_tile_row, _dev_props, _info->fp_type, _info->opt_args,
          _info->profile_type, _stream);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev, _T_B_dev, _T_A_dev);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
      break;
  }
  return error;
}

SCAMPError_t Tile::do_self_join_half() {
  SCAMPError_t error = SCAMP_NO_ERROR;

  if (_info->mp_window > _current_tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (_info->mp_window > _current_tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          _scratch->compute_QT(_QT_dev, _T_A_dev, _T_B_dev, _means_B, _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_upper(
          _QT_dev, _df_A, _df_B, _dg_A, _dg_B, _norms_A, _norms_B,
          &_profile_a_tile_dev, &_profile_b_tile_dev, _info->mp_window,
          _current_tile_width - _info->mp_window + 1,
          _current_tile_height - _info->mp_window + 1, _current_tile_col,
          _current_tile_row, _dev_props, _info->fp_type, _info->opt_args,
          _info->profile_type, _stream);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev, _T_A_dev, _T_B_dev);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_self_join_upper(this);
      break;
  }
  if (error != SCAMP_NO_ERROR) {
    return error;
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t Tile::do_ab_join_full() {
  SCAMPError_t error = SCAMP_NO_ERROR;

  if (_info->mp_window > _current_tile_width) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (_info->mp_window > _current_tile_height) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          _scratch->compute_QT(_QT_dev, _T_A_dev, _T_B_dev, _means_B, _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_ab_join_upper(
          _QT_dev, _df_A, _df_B, _dg_A, _dg_B, _norms_A, _norms_B,
          &_profile_a_tile_dev, &_profile_b_tile_dev, _info->mp_window,
          _current_tile_width - _info->mp_window + 1,
          _current_tile_height - _info->mp_window + 1, _current_tile_col,
          _current_tile_row, _info->global_start_col_position,
          _info->global_start_row_position, _info->is_aligned, _dev_props,
          _info->fp_type, _info->computing_rows, _info->opt_args,
          _info->profile_type, _stream);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev, _T_A_dev, _T_B_dev);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_ab_join_upper(this);
      break;
  }
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          _scratch->compute_QT(_QT_dev, _T_B_dev, _T_A_dev, _means_A, _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_ab_join_lower(
          _QT_dev, _df_A, _df_B, _dg_A, _dg_B, _norms_A, _norms_B,
          &_profile_a_tile_dev, &_profile_b_tile_dev, _info->mp_window,
          _current_tile_width - _info->mp_window + 1,
          _current_tile_height - _info->mp_window + 1, _current_tile_col,
          _current_tile_row, _info->global_start_col_position,
          _info->global_start_row_position, _info->is_aligned, _dev_props,
          _info->fp_type, _info->computing_rows, _info->opt_args,
          _info->profile_type, _stream);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev, _T_B_dev, _T_A_dev);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_ab_join_lower(this);
      break;
  }
  return error;
}

}  // namespace SCAMP
