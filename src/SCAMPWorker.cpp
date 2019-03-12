#include "SCAMPWorker.h"

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

void Worker::Init() {
  switch(_arch) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      cudaSetDevice(_cuda_id);
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      break;
  }
}

void Worker::Sync() {
  switch(_arch) {
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

void Worker::InitTimeseries(
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h) {

  switch(_arch) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      gpuErrchk(cudaPeekAtLastError());
      printf("width = %lu\n", _current_tile_width);
      printf("height = %lu\n", _current_tile_height);
      cudaMemcpyAsync(_T_A_dev, Ta_h.data() + _current_tile_col,
                  sizeof(double) * _current_tile_width,
                  cudaMemcpyHostToDevice, _stream);
      printf("hello\n");
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_T_B_dev, Tb_h.data() + _current_tile_row,
                  sizeof(double) * _current_tile_height,
                  cudaMemcpyHostToDevice, _stream);
      printf("hello2\n");
      gpuErrchk(cudaPeekAtLastError());
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      // FIXME: stub
      //_T_A_dev = Ta_h.data() + _current_tile_col;
      //_T_B_dev = Tb_h.data() + _current_tile_row;
      break;
  }
}


Profile Worker::AllocProfile(SCAMPProfileType t, uint64_t size) {
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

SCAMPError_t Worker::InitProfile(Profile *profile_a, Profile *profile_b, bool self_join, bool computing_rows, bool keep_rows) {
  int profile_size = GetProfileTypeSize(_profile_type);
  int width = _current_tile_width - _mp_window + 1;
  int height = _current_tile_height - _mp_window + 1;

  switch(_arch) {
    case CUDA_GPU_WORKER: {
#if _HAS_CUDA_
    switch (_profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        cudaMemsetAsync(_profile_a_tile_dev.at(_profile_type), 0, profile_size * width, _stream);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemsetAsync(_profile_b_tile_dev.at(_profile_type), 0, profile_size * height, _stream);
        gpuErrchk(cudaPeekAtLastError());
        break;
      case PROFILE_TYPE_1NN_INDEX: {
        const uint64_t *pA_ptr = profile_a->data().Get(0).uint64_value().value().data();
        cudaMemcpyAsync(_profile_a_tile_dev.at(_profile_type),  pA_ptr + _current_tile_col, sizeof(uint64_t) * width, cudaMemcpyHostToDevice, _stream);
        gpuErrchk(cudaPeekAtLastError());
        if (self_join) {
          cudaMemcpyAsync(_profile_b_tile_dev.at(_profile_type), pA_ptr + _current_tile_row, sizeof(uint64_t) * height, cudaMemcpyHostToDevice, _stream);
          gpuErrchk(cudaPeekAtLastError());

        } else if (computing_rows && keep_rows) {
          const uint64_t *pB_ptr =
              profile_b->data().Get(0).uint64_value().value().data();
          cudaMemcpyAsync(
              _profile_b_tile_dev.at(_profile_type),
              pB_ptr + _current_tile_row,
              sizeof(uint64_t) * height,
              cudaMemcpyHostToDevice, _stream);
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
    case CPU_WORKER:
      switch(_profile_type) {
        case PROFILE_TYPE_1NN_INDEX:
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

void Worker::InitStats(const PrecomputedInfo& a, const PrecomputedInfo& b) {
  size_t bytes_a =
      (_current_tile_width - _mp_window + 1) * sizeof(double);
  size_t bytes_b =
      (_current_tile_height - _mp_window + 1) * sizeof(double);
  switch(_arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaMemcpyAsync(_norms_A, a.norms().data() + _current_tile_col, bytes_a, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_df_A, a.df().data() + _current_tile_col, bytes_a, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_dg_A, a.dg().data() + _current_tile_col, bytes_a, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_means_A, a.means().data() + _current_tile_col, bytes_a, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_norms_B, b.norms().data() + _current_tile_row, bytes_b, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_df_B, b.df().data() + _current_tile_row, bytes_b, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_dg_B, b.dg().data() + _current_tile_row, bytes_b, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
      cudaMemcpyAsync(_means_B, b.means().data() + _current_tile_row, bytes_b, cudaMemcpyHostToDevice, _stream);
      gpuErrchk(cudaPeekAtLastError());
#else
      assert("ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      /*
      FIXME: stub
      _norms_A = a.norms().data() + _current_tile_col;
      _df_A = a.df().data() + _current_tile_col;
      _dg_A = a.dg().data() + _current_tile_col;
      _means_A = a.means().data() + _current_tile_col;
      _norms_B = b.norms().data() + _current_tile_row;
      _df_B = b.df().data() + _current_tile_row;
      _dg_B = b.dg().data() + _current_tile_row;
      _means_B = b.means().data() + _current_tile_row;
      */
     break; 
  }
}

void Worker::MergeTileIntoFullProfile(Profile *tile_profile,
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

void Worker::MergeProfile(bool self_join, bool fetch_rows, bool keep_rows, Profile *profile_a, std::mutex &a_lock,  Profile *profile_b, std::mutex &b_lock) {
  // Set up a copy operation back to the host
  CopyProfileToHost(&_profile_a_tile, &_profile_a_tile_dev,
                      _current_tile_width - _mp_window + 1);
  if (fetch_rows) {
    CopyProfileToHost(&_profile_b_tile, &_profile_b_tile_dev,
                        _current_tile_height - _mp_window + 1);
  }

  // Wait for the previous work to be done
  Sync();
  
  // Merge result
  MergeTileIntoFullProfile(&_profile_a_tile, _current_tile_col, _current_tile_width - _mp_window + 1, profile_a, _current_tile_row, a_lock);
  if (self_join) {
    MergeTileIntoFullProfile(&_profile_b_tile, _current_tile_row, _current_tile_height - _mp_window + 1, profile_a, _current_tile_col, a_lock);
  } else if (fetch_rows && keep_rows) {
    MergeTileIntoFullProfile(&_profile_b_tile, _current_tile_row, _current_tile_height - _mp_window + 1, profile_b, _current_tile_col, b_lock);
  }


}
// TODO(zpzim): make CPU/GPU agnostic
void Worker::CopyProfileToHost(Profile *destination_profile, const DeviceProfile *device_tile_profile,
    uint64_t length) {
  switch (_arch) {
    case CUDA_GPU_WORKER: {
#ifdef _HAS_CUDA_
      switch (_profile_type) {
        case PROFILE_TYPE_SUM_THRESH:
          cudaMemcpyAsync(destination_profile->mutable_data()
                              ->Mutable(0)
                              ->mutable_double_value()
                              ->mutable_value()
                              ->mutable_data(),
                          device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
                          length * sizeof(double), cudaMemcpyDeviceToHost, _stream);
          gpuErrchk(cudaPeekAtLastError());
          break;
        case PROFILE_TYPE_1NN_INDEX:
          cudaMemcpyAsync(destination_profile->mutable_data()
                              ->Mutable(0)
                              ->mutable_uint64_value()
                              ->mutable_value()
                              ->mutable_data(),
                          device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
                          length * sizeof(uint64_t), cudaMemcpyDeviceToHost, _stream);
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
      // TODO: implement stub
      break;
  }
}


void Worker::init_cuda() {
#ifdef _HAS_CUDA_
  cudaSetDevice(_cuda_id);
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&_dev_props, _cuda_id);
  cudaStreamCreate(&_stream);
  size_t profile_size = GetProfileTypeSize(_profile_type);
  cudaMalloc(&_T_A_dev, sizeof(double) * _max_tile_ts_size);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_T_B_dev, sizeof(double) * _max_tile_ts_size);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_profile_a_tile_dev.at(_profile_type),
           profile_size * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_profile_b_tile_dev.at(_profile_type),
           profile_size * _max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_QT_dev, sizeof(double) * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_means_A, sizeof(double) * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_means_B, sizeof(double) * _max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_norms_A, sizeof(double) * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_norms_B, sizeof(double) * _max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_df_A, sizeof(double) * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_df_B, sizeof(double) * _max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_dg_A, sizeof(double) * _max_tile_width);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_dg_B, sizeof(double) * _max_tile_height);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&_scratchpad, sizeof(double) * _max_tile_ts_size);
  _scratch = std::make_shared<fft_precompute_helper>(
      _max_tile_ts_size, _mp_window, true);
#else
  assert("ERROR: CUDA used in binary not built with CUDA");
#endif
}

void Worker::free_cuda() {
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
  cudaFree(_profile_a_tile_dev.at(_profile_type));
  cudaFree(_profile_b_tile_dev.at(_profile_type));
  cudaFree(_scratchpad);
  cudaStreamDestroy(_stream);
#else
  assert("ERROR: CUDA used in binary not built with CUDA");
#endif 
}

// TODO(zpzim): Finish STUB
void Worker::init_cpu() {

}


// TODO(zpzim): Finish STUB
void Worker::free_cpu() {

}


} // namespace SCAMP
