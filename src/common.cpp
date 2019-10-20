#include "common.h"
#include "scamp_exception.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace SCAMP {

void Memcopy(void *destination, const void *source, size_t bytes,
             bool from_tile, const ExecInfo *info) {
  switch (info->arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(info->cuda_id);
      gpuErrchk(cudaPeekAtLastError());
      if (from_tile) {
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost,
                        info->stream);
      } else {
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice,
                        info->stream);
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

void match_merge(const std::vector<SCAMPmatch> &matches, ProfileData *profile,
                 uint64_t merge_start_row, uint64_t merge_start_col,
                 int64_t max_matches) {
  for (auto elem : matches) {
    uint64_t col = elem.col + merge_start_col;
    auto &pq = profile->match_value[col];
    if (pq.size() == max_matches && pq.top().corr < elem.corr) {
      pq.pop();
      elem.col = col;
      elem.row += merge_start_row;
      pq.push(elem);
    } else if (pq.size() < max_matches) {
      elem.col = col;
      elem.row += merge_start_row;
      pq.push(elem);
    }
  }
}

void Profile::Alloc(size_t size) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      data.emplace_back();
      data[0].double_value.resize(size, 0);
      break;
    case PROFILE_TYPE_1NN:
      data.emplace_back();
      data[0].float_value.resize(size, std::numeric_limits<float>::lowest());
      break;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = std::numeric_limits<float>::lowest();
      data.emplace_back();
      data[0].uint64_value.resize(size, e.ulong);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      data.emplace_back();
      data[0].uint64_value.resize(size, 0);
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      data.emplace_back();
      data[0].match_value.resize(size);
      break;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      break;
  }
}

// Copies a profile to the host
void Profile::CopyFromDevice(const ExecInfo *info,
                             const DeviceProfile *device_tile_profile,
                             uint64_t length) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memcopy(this->data[0].double_value.data(),
              device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
              length * sizeof(double), true, info);
      break;
    case PROFILE_TYPE_1NN:
      Memcopy(this->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN), length * sizeof(float),
              true, info);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      Memcopy(this->data[0].uint64_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
              length * sizeof(uint64_t), true, info);
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      this->data[0].match_value_unordered.resize(length);
      Memcopy(this->data[0].match_value_unordered.data(),
              device_tile_profile->at(PROFILE_TYPE_APPROX_ALL_NEIGHBORS),
              length * sizeof(SCAMPmatch), true, info);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      break;
  }
}

// Merges a profile corresponding to the result of a tile into this profile
void Profile::MergeTileToProfile(Profile *tile_profile, const OpInfo *info,
                                 uint64_t position, uint64_t length,
                                 uint64_t index_start) {
  // Lock the before we merge, this function can be called by multiple threads
  std::unique_lock<std::mutex> mlock(this->_profile_lock);
  if (type != tile_profile->type) {
    throw(SCAMPException("Profile Types do not match"));
  }
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(this->data[0].double_value.data(), position,
                              length,
                              tile_profile->data[0].double_value.data());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(
          this->data[0].uint64_value.data(), position, length,
          tile_profile->data[0].uint64_value.data(), index_start);
      return;
    case PROFILE_TYPE_1NN:
      elementwise_max<float>(this->data[0].float_value.data(), position, length,
                             tile_profile->data[0].float_value.data());
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<uint64_t>(this->data[0].uint64_value.data(), position,
                                length,
                                tile_profile->data[0].uint64_value.data());
      return;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      match_merge(tile_profile->data[0].match_value_unordered, &this->data[0],
                  index_start, position, info->max_matches_per_column);
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      return;
  }
}

void SCAMPArgs::validate() {
  if (window < 3) {
    throw SCAMPException("Error: Subsequence length must be at least 3");
  }
  if (max_tile_size < 1024) {
    throw SCAMPException("Error: max tile size must be at least 1024");
  }
  if (max_tile_size / 2 < window) {
    throw SCAMPException(
        "Error: Tile length and width must be at least 2x larger than the "
        "window size");
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    throw SCAMPException(
        "Error: Input time series must be at least 'subesequence window size' "
        "in length");
  }
}

std::string GetProfileTypeString(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_INVALID:
      return "PROFILE_TYPE_INVALID";
    case PROFILE_TYPE_1NN_INDEX:
      return "PROFILE_TYPE_1NN_INDEX";
    case PROFILE_TYPE_1NN:
      return "PROFILE_TYPE_1NN";
    case PROFILE_TYPE_SUM_THRESH:
      return "PROFILE_TYPE_SUM_THRESH";
    case PROFILE_TYPE_FREQUENCY_THRESH:
      return "PROFILE_TYPE_FREQUENCY_THRESH";
    case PROFILE_TYPE_KNN:
      return "PROFILE_TYPE_KNN";
    case PROFILE_TYPE_1NN_MULTIDIM:
      return "PROFILE_TYPE_1NN_MULTIDIM";
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return "PROFILE_TYPE_APPROX_ALL_NEIGHBORS";
  }
}

std::string GetPrecisionTypeString(SCAMPPrecisionType t) {
  switch (t) {
    case PRECISION_INVALID:
      return "PRECISION_INVALID";
    case PRECISION_SINGLE:
      return "PRECISION_SINGLE";
    case PRECISION_MIXED:
      return "PRECISION_MIXED";
    case PRECISION_DOUBLE:
      return "PRECISION_DOUBLE";
  }
}

std::string getSCAMPErrorString(SCAMPError_t err) {
  switch (err) {
    case SCAMP_NO_ERROR:
      return "SCAMP_NO_ERROR";
    case SCAMP_FUNCTIONALITY_UNIMPLEMENTED:
      return "SCAMP_FUNCTIONALITY_UNIMPLEMENTED";
    case SCAMP_TILE_ILLEGAL_TYPE:
      return "SCAMP_TILE_ILLEGAL_TYPE";
    case SCAMP_CUDA_ERROR:
      return "SCAMP_CUDA_ERROR";
    case SCAMP_CUFFT_ERROR:
      return "SCAMP_CUFFT_ERROR";
    case SCAMP_CUFFT_EXEC_ERROR:
      return "SCAMP_CUFFT_EXEC_ERROR";
    case SCAMP_DIM_INCOMPATIBLE:
      return "SCAMP_DIM_INCOMPATIBLE";
  }
}

void SCAMPArgs::print() {
  std::cout << "window: " << window << std::endl;
  std::cout << "max_tile_size: " << max_tile_size << std::endl;
  std::cout << "has_b: " << has_b << std::endl;
  std::cout << "keep_rows_separate: " << keep_rows_separate << std::endl;
  std::cout << "distributed_start_row: " << distributed_start_row << std::endl;
  std::cout << "distributed_start_col: " << distributed_start_col << std::endl;
  std::cout << "computing_rows: " << computing_rows << std::endl;
  std::cout << "computing_columns: " << computing_columns << std::endl;
  std::cout << "is_aligned: " << is_aligned << std::endl;
  std::cout << "profile_type: " << GetProfileTypeString(profile_type)
            << std::endl;
  std::cout << "precision_type: " << GetPrecisionTypeString(precision_type)
            << std::endl;
  std::cout << "distance_threshold: " << distance_threshold << std::endl;
  std::cout << "silent_mode: " << silent_mode << std::endl;
  std::cout << "max_matches_per_column: " << max_matches_per_column
            << std::endl;
  std::cout << "timeseries_a size: " << timeseries_a.size() << std::endl;
  std::cout << "timeseries_b size: " << timeseries_b.size() << std::endl;
}

size_t GetProfileTypeSize(SCAMPProfileType t) {
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      return sizeof(double);
    case PROFILE_TYPE_1NN_INDEX:
      return sizeof(uint64_t);
    case PROFILE_TYPE_1NN:
      return sizeof(float);
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_KNN:
      return sizeof(SCAMPmatch);
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_INVALID:
    default:
      throw SCAMPException(
          "Error: Could not determine size of profile elements");
  }
}

}  // namespace SCAMP

#ifdef _HAS_CUDA_
void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::ostringstream ostream;
    ostream << "GPUasssert: " << cudaGetErrorString(code) << " " << file << " "
            << line;
    throw SCAMPException(ostream.str());
  }
}
#endif
