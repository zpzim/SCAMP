#include "tile.h"
#include <functional>
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

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    if (mp_full[i + merge_start] < to_merge[i]) {
      mp_full[i + merge_start] = to_merge[i];
    }
  }
}

// Gets the exclusion zone for a particular tile (helper_function)
static int get_exclusion(uint64_t window_size, int64_t start_row,
                         int64_t start_column) {
  int exclusion = window_size / 4;
  if (start_column >= start_row && start_column < start_row + exclusion) {
    return exclusion;
  }
  return 0;
}

std::pair<int, int> Tile::get_exclusion_for_self_join(bool upper_tile) {
  int exclusion;
  if (upper_tile) {
    exclusion = get_exclusion(_info->mp_window, get_tile_row(), get_tile_col());
    return std::make_pair(exclusion, 0);
  }
  size_t height = get_tile_height() - _info->mp_window + 1;
  exclusion =
      get_exclusion(_info->mp_window, get_tile_col(), get_tile_row() + height);
  return std::make_pair(0, exclusion);
}

// Gets the exclusion zone for a particular tile (logic for ab joins)
std::pair<int, int> Tile::get_exclusion_for_ab_join(bool upper_tile) {
  int exclusion_lower = 0;
  int exclusion_upper = 0;

  if (!_info->is_aligned) {
    return std::make_pair(exclusion_lower, exclusion_upper);
  }
  size_t height = get_tile_height() - _info->mp_window + 1;
  size_t width = get_tile_width() - _info->mp_window + 1;

  int start_col = get_tile_col();
  int start_row = get_tile_row();
  if (_info->global_start_col_position >= 0 &&
      _info->global_start_row_position >= 0) {
    start_col += _info->global_start_col_position;
    start_row += _info->global_start_row_position;
  }
  if (upper_tile) {
    exclusion_lower = get_exclusion(_info->mp_window, start_row, start_col);
    if (start_row > start_col) {
      exclusion_upper =
          get_exclusion(_info->mp_window, start_row, start_col + width);
    } else {
      exclusion_upper = 0;
    }
  } else {
    exclusion_lower = get_exclusion(_info->mp_window, start_col, start_row);
    if (start_row >= start_col) {
      exclusion_upper = 0;
    } else {
      exclusion_upper =
          get_exclusion(_info->mp_window, start_col, start_row + height);
    }
  }
  return std::make_pair(exclusion_lower, exclusion_upper);
}

// Allocator for tile memory which can reside on the host or cuda devices
template <typename T>
T *alloc_mem(size_t count, SCAMPArchitecture arch, int deviceid) {
  switch (arch) {
    case CUDA_GPU_WORKER: {
#ifdef _HAS_CUDA_
      cudaSetDevice(deviceid);
      size_t bytes = count * sizeof(T);
      T *ptr;
      cudaMalloc(&ptr, bytes);
      gpuErrchk(cudaPeekAtLastError());
      return ptr;
#else
      ASSERT(false, "Using CUDA in binary not built with it");
      return nullptr;
#endif
    }
    case CPU_WORKER:
      return new T[count];  // NOLINT
  }
}

// Deleter for tile memory which can reside on the host or cuda devices
template <typename T>
void free_mem(T *ptr, SCAMPArchitecture arch, int deviceid) {
  switch (arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(deviceid);
      cudaFree(ptr);
#else
      ASSERT(false, "Using CUDA in binary not built with it");
#endif
      break;
    case CPU_WORKER:
      delete[] ptr;  // NOLINT
      break;
  }
}

void Tile::Memset(void *destination, char value, size_t bytes) {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(_cuda_id);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemsetAsync(destination, value, bytes, _stream);
      gpuErrchk(cudaPeekAtLastError());
#else
      ASSERT(false, "Using CUDA in binary not built with it");
#endif
      break;
    case CPU_WORKER:
      memset(destination, value, bytes);
      break;
  }
}

void Tile::Memcopy(void *destination, const void *source, size_t bytes,
                   bool from_tile) {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(_cuda_id);
      gpuErrchk(cudaPeekAtLastError());
      if (from_tile) {
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost,
                        _stream);
      } else {
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice,
                        _stream);
      }
      gpuErrchk(cudaPeekAtLastError());
#else
      ASSERT(false, "Using CUDA in binary not built with it");
#endif
      break;
    case CPU_WORKER:
      // TODO(zpzim): Most of the time we don't actually have to copy
      // memory here, we can just set a reference.
      memcpy(destination, source, bytes);
      break;
  }
}

Tile::Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id)
    : _info(info),
      _arch(arch),
      _cuda_id(cuda_id),
      _current_tile_width(0),
      _current_tile_height(0),
      _current_tile_row(0),
      _current_tile_col(0),
      // Allocate memory based on architecture
      _T_A_dev(alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _T_B_dev(alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _QT_dev(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
              [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _means_A(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _means_B(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _norms_A(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _norms_B(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _df_A(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _df_B(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _dg_A(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _dg_B(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      _scratchpad(
          static_cast<double *>(
              alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id)),
          [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),

      _scratch(std::unique_ptr<qt_compute_helper>(new qt_compute_helper(
          info->max_tile_ts_size, info->mp_window, true, arch)))
#ifdef _HAS_CUDA_
      ,
      _stream(),
      _dev_props()
#endif
{
  size_t profile_size = GetProfileTypeSize(_info->profile_type);
  _profile_a_tile_dev[_info->profile_type] =
      alloc_mem<char>(profile_size * _info->max_tile_width, arch, cuda_id);
  _profile_b_tile_dev[_info->profile_type] =
      alloc_mem<char>(profile_size * _info->max_tile_height, arch, cuda_id);

  _profile_a_tile = AllocProfile(_info->profile_type, _info->max_tile_height);
  _profile_b_tile = AllocProfile(_info->profile_type, _info->max_tile_width);

  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(_cuda_id);
      cudaGetDeviceProperties(&_dev_props, _cuda_id);
      cudaStreamCreate(&_stream);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      // Add any arch-specific inits here
      break;
  }
}

Tile::~Tile() {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(_cuda_id);
      cudaStreamDestroy(_stream);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      // Add any arch-specific cleanup here
      break;
  }
  free_mem<char>(static_cast<char *>(_profile_a_tile_dev[_info->profile_type]),
                 _arch, _cuda_id);
  free_mem<char>(static_cast<char *>(_profile_b_tile_dev[_info->profile_type]),
                 _arch, _cuda_id);
}

void Tile::Sync() {
  switch (_arch) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      cudaStreamSynchronize(_stream);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      break;
  }
}

void Tile::InitTimeseries(const std::vector<double> &Ta_h,
                          const std::vector<double> &Tb_h) {
  Memcopy(_T_A_dev.get(), Ta_h.data() + _current_tile_col,
          sizeof(double) * _current_tile_width, false);
  Memcopy(_T_B_dev.get(), Tb_h.data() + _current_tile_row,
          sizeof(double) * _current_tile_height, false);
}

Profile Tile::AllocProfile(SCAMPProfileType t, uint64_t size) {
  Profile p;
  p.type = t;
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      p.data.emplace_back();
      p.data[0].double_value.resize(size, 0);
      return p;
    case PROFILE_TYPE_1NN:
      p.data.emplace_back();
      p.data[0].float_value.resize(size, std::numeric_limits<float>::lowest());
      return p;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = std::numeric_limits<float>::lowest();
      p.data.emplace_back();
      p.data[0].uint64_value.resize(size, e.ulong);
      return p;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      p.data.emplace_back();
      p.data[0].uint64_value.resize(size, 0);
      return p;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return p;
  }
}

// Initializes the tile's local profile values based on global profiles
// "profile_a" and "profile_b"
SCAMPError_t Tile::InitProfile(Profile *profile_a, Profile *profile_b) {
  int profile_size = GetProfileTypeSize(_info->profile_type);
  int width = _current_tile_width - _info->mp_window + 1;
  int height = _current_tile_height - _info->mp_window + 1;
  SCAMPProfileType type = _info->profile_type;
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memset(_profile_a_tile_dev.at(type), 0, profile_size * width);
      Memset(_profile_b_tile_dev.at(type), 0, profile_size * height);
      break;
    case PROFILE_TYPE_1NN_INDEX: {
      const uint64_t *pA_ptr = profile_a->data[0].uint64_value.data();
      Memcopy(_profile_a_tile_dev.at(type), pA_ptr + _current_tile_col,
              sizeof(uint64_t) * width, false);
      if (_info->self_join) {
        Memcopy(_profile_b_tile_dev.at(type), pA_ptr + _current_tile_row,
                sizeof(uint64_t) * height, false);
      } else if (_info->computing_rows && _info->keep_rows_separate) {
        const uint64_t *pB_ptr = profile_b->data[0].uint64_value.data();
        Memcopy(_profile_b_tile_dev.at(type), pB_ptr + _current_tile_row,
                sizeof(uint64_t) * height, false);
      }
      break;
    }
    case PROFILE_TYPE_1NN: {
      const float *pA_ptr = profile_a->data[0].float_value.data();
      Memcopy(_profile_a_tile_dev.at(type), pA_ptr + _current_tile_col,
              sizeof(float) * width, false);
      if (_info->self_join) {
        Memcopy(_profile_b_tile_dev.at(type), pA_ptr + _current_tile_row,
                sizeof(float) * height, false);
      } else if (_info->computing_rows && _info->keep_rows_separate) {
        const float *pB_ptr = profile_b->data[0].float_value.data();
        Memcopy(_profile_b_tile_dev.at(type), pB_ptr + _current_tile_row,
                sizeof(float) * height, false);
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

// Initialize the tile's local stats based on global statistics "a" and "b"
void Tile::InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b) {
  size_t bytes_a =
      (_current_tile_width - _info->mp_window + 1) * sizeof(double);
  size_t bytes_b =
      (_current_tile_height - _info->mp_window + 1) * sizeof(double);
  Memcopy(_norms_A.get(), a.norms().data() + _current_tile_col, bytes_a, false);
  Memcopy(_df_A.get(), a.df().data() + _current_tile_col, bytes_a, false);
  Memcopy(_dg_A.get(), a.dg().data() + _current_tile_col, bytes_a, false);
  Memcopy(_means_A.get(), a.means().data() + _current_tile_col, bytes_a, false);
  Memcopy(_norms_B.get(), b.norms().data() + _current_tile_row, bytes_b, false);
  Memcopy(_df_B.get(), b.df().data() + _current_tile_row, bytes_b, false);
  Memcopy(_dg_B.get(), b.dg().data() + _current_tile_row, bytes_b, false);
  Memcopy(_means_B.get(), b.means().data() + _current_tile_row, bytes_b, false);
}

// TODO(zpzim): move this back into SCAMP_Operation, we shouldn't have the
// merging be functionality of the individual tile
// Merges a local result "tile_profile" with the global matrix profile
// "full_profile"
void Tile::MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                    uint64_t length, Profile *full_profile,
                                    uint64_t index_start, std::mutex &lock) {
  // Lock the entire result vector before we merge
  // TODO(zpzim): we don't have to do this, we only need to lock the specific
  // "tile row" or "tile_column" that we are updating
  std::unique_lock<std::mutex> mlock(lock);
  switch (_info->profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->data[0].double_value.data(),
                              position, length,
                              tile_profile->data[0].double_value.data());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(
          full_profile->data[0].uint64_value.data(), position, length,
          tile_profile->data[0].uint64_value.data(), index_start);
      return;
    case PROFILE_TYPE_1NN:
      elementwise_max<float>(full_profile->data[0].float_value.data(), position,
                             length, tile_profile->data[0].float_value.data());
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<uint64_t>(full_profile->data[0].uint64_value.data(),
                                position, length,
                                tile_profile->data[0].uint64_value.data());
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

// TODO(zpzim): move this back into SCAMP_Operation, we shouldn't have the
// merging be functionality of the individual tile
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

// Copies a profile to the host
void Tile::CopyProfileToHost(Profile *destination_profile,
                             const DeviceProfile *device_tile_profile,
                             uint64_t length) {
  switch (_info->profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memcopy(destination_profile->data[0].double_value.data(),
              device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
              length * sizeof(double), true);
      break;
    case PROFILE_TYPE_1NN:
      Memcopy(destination_profile->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN), length * sizeof(float),
              true);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      Memcopy(destination_profile->data[0].uint64_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
              length * sizeof(uint64_t), true);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      break;
  }
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
      error = _scratch->compute_QT(_QT_dev.get(), _T_B_dev.get(),
                                   _T_A_dev.get(), _means_A.get(), _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_lower(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev.get(), _T_B_dev.get(),
                                       _T_A_dev.get());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_self_join_lower(this);
      break;
  }
  return error;
}

// Computes the matrix profile upper triangular portion of the tile
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
      error = _scratch->compute_QT(_QT_dev.get(), _T_A_dev.get(),
                                   _T_B_dev.get(), _means_B.get(), _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_upper(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev.get(), _T_A_dev.get(),
                                       _T_B_dev.get());
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
      error = _scratch->compute_QT(_QT_dev.get(), _T_A_dev.get(),
                                   _T_B_dev.get(), _means_B.get(), _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_ab_join_upper(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev.get(), _T_A_dev.get(),
                                       _T_B_dev.get());
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
      error = _scratch->compute_QT(_QT_dev.get(), _T_B_dev.get(),
                                   _T_A_dev.get(), _means_A.get(), _stream);
      if (error != SCAMP_NO_ERROR) {
        return error;
      }

      error = gpu_kernel_ab_join_lower(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = _scratch->compute_QT_CPU(_QT_dev.get(), _T_B_dev.get(),
                                       _T_A_dev.get());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_ab_join_lower(this);
      break;
  }
  return error;
}

}  // namespace SCAMP
