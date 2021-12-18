#include "common.h"
#include "scamp_exception.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace SCAMP {

static constexpr int64_t GIGABYTE = 1024 * 1024 * 1024;

static constexpr int64_t MEMORY_SAVINGS_FACTOR = 200;

bool NeedsSort(SCAMPProfileType type) {
  return type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
}

bool NeedsIntermittentMerge(SCAMPProfileType type) {
  return type != PROFILE_TYPE_MATRIX_SUMMARY;
}

bool NeedsIntermittentReset(SCAMPProfileType type) {
  return type != PROFILE_TYPE_MATRIX_SUMMARY;
}

// TODO(zpzim): make this a more generic parameter that is specified by
// the user or memory availibility
static constexpr int64_t PROFILE_MEMORY_BUDGET = 0.5 * GIGABYTE;

OpInfo::OpInfo(size_t Asize, size_t Bsize, size_t window_sz,
               size_t max_tile_size, bool selfjoin, SCAMPPrecisionType t,
               int64_t start_row, int64_t start_col, OptionalArgs args_,
               SCAMPProfileType profiletype, bool keep_rows, bool compute_rows,
               bool compute_cols, bool aligned, bool silent_mode,
               int num_workers, int64_t max_matches_per_col, int64_t mheight,
               int64_t mwidth)
    : full_ts_len_A(Asize),
      full_ts_len_B(Bsize),
      mp_window(window_sz),
      self_join(selfjoin),
      fp_type(t),
      global_start_row_position(start_row),
      global_start_col_position(start_col),
      opt_args(args_),
      profile_type(profiletype),
      keep_rows_separate(keep_rows),
      computing_rows(compute_rows),
      computing_cols(compute_cols),
      is_aligned(aligned),
      silent_mode(silent_mode),
      max_matches_per_column(max_matches_per_col),
      matrix_height(mheight),
      matrix_width(mwidth) {
  if (self_join) {
    full_ts_len_B = full_ts_len_A;
  }
  auto maxSize = std::max(Asize, Bsize);
  max_tile_ts_size = maxSize / (num_workers);

  if (max_tile_ts_size > max_tile_size) {
    max_tile_ts_size = max_tile_size;
  }

  // Prevents our tiles from becoming pathalogically small
  // Tiles should not be smaller than the exclusion zone (mp_window / 4)
  // otherwise the tiling becomes unnecessarially complex
  const int SMALLEST_ALLOWED_TILE_DIM = mp_window;

  if (max_tile_ts_size < SMALLEST_ALLOWED_TILE_DIM + mp_window) {
    max_tile_ts_size = SMALLEST_ALLOWED_TILE_DIM + mp_window;
  }

  max_tile_width = max_tile_ts_size - mp_window + 1;
  max_tile_height = max_tile_width;

  int64_t normative_match_budget_per_tile =
      (PROFILE_MEMORY_BUDGET / num_workers) / sizeof(SCAMPmatch);

  max_matches_per_tile = max_matches_per_column * max_tile_width;

  if (normative_match_budget_per_tile > max_matches_per_tile) {
    max_matches_per_tile = normative_match_budget_per_tile;
  }

  int64_t worker_memory_budget = max_matches_per_tile * sizeof(SCAMPmatch) * 2;

  if (!silent_mode && profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    std::cout << "Have to allocate space for " << max_matches_per_tile
              << " matches per tile, which will require on the order of "
              << worker_memory_budget / static_cast<double>(GIGABYTE)
              << " GB of memory per worker.";
    std::cout << "If this amount of memory is too large we may run out of "
                 "memory on the system/GPUs, if this happens try reducing "
                 "max_matches_per_column to a smaller value.";
  }

  // Matrix summaries only need to reduce along the columns.
  if (profile_type == PROFILE_TYPE_MATRIX_SUMMARY) {
    computing_rows = false;
    keep_rows_separate = false;
    cols_per_cell = std::ceil((full_ts_len_A - mp_window + 1) /
                              static_cast<double>(matrix_width));
    rows_per_cell = std::ceil((full_ts_len_B - mp_window + 1) /
                              static_cast<double>(matrix_height));
  }
}

ExecInfo::ExecInfo(SCAMPArchitecture _arch, int _cuda_id)
    : arch(_arch), cuda_id(_cuda_id) {
  switch (arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(cuda_id);
      cudaGetDeviceProperties(&dev_props, cuda_id);
      cudaStreamCreate(&stream);
#else
      ASSERT(false, "Binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      break;
  }
}

ExecInfo::~ExecInfo() {
  switch (arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      cudaSetDevice(cuda_id);
      cudaStreamDestroy(stream);
#endif
      break;
    case CPU_WORKER:
      // Add any arch-specific cleanup here
      break;
  }
}

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

// Updates the adaptive thresholds of a tile in the case that there was an
// overflow. This is very similar logic to match_merge, except it does not
// add any elements to the top K lists and only updates the thresholds.
void Profile::threshold_merge(const std::vector<SCAMPmatch> &matches,
                              uint64_t merge_start_col, int64_t max_matches) {
  if (matches.empty()) {
    return;
  }

  int i = 0;
  while (i < matches.size()) {
    uint64_t curr_col = matches[i].col;
    int count = 1;
    // Count how many results we have for the current column.
    while (i + count < matches.size() && matches[i + count].col == curr_col) {
      ++count;
    }
    // If we have more than max_matches we can update the threshold with the
    // smallest value.
    if (count - 1 > max_matches && thresholds[curr_col + merge_start_col] <
                                       matches[i + max_matches].corr) {
      thresholds[curr_col + merge_start_col] = matches[i + max_matches].corr;
    }
    i += count;
  }
}

// Merges the elements in matches into the top K values in the current
// profile. Matches must be properly sorted first by column in ascending
// order, then by correlation in descending order.
void Profile::match_merge(const std::vector<SCAMPmatch> &matches,
                          uint64_t merge_start_row, uint64_t merge_start_col,
                          int64_t max_matches) {
  if (matches.empty()) {
    return;
  }

  int i = 0;
  while (i < matches.size()) {
    uint64_t curr_col = matches[i].col;
    auto &pq = this->data.front().match_value[curr_col + merge_start_col];
    uint64_t count = 0;
    float old_val;
    bool update_possible = false;
    // Loop over the initial values that might need to go into the top K
    while (i + count < matches.size() && matches[i + count].col == curr_col &&
           count < max_matches) {
      auto &match = matches[i + count];
      // If the match is not better than the bottom of the top K, break out.
      if (pq.size() == max_matches && match.corr <= pq.top().corr) {
        break;
      }
      // If we have found K values we need to make some space for the new one.
      if (pq.size() == max_matches) {
        update_possible = true;
        old_val = pq.top().corr;
        pq.pop();
      }
      pq.emplace(match.corr, match.row + merge_start_row,
                 match.col + merge_start_col);
      ++count;
    }

    // Skip the rest of the values for this column, they aren't useful.
    while (i + count < matches.size() && matches[i + count].col == curr_col) {
      ++count;
    }
    // If we ever updated the top K we can update the threshold.
    if (update_possible) {
      this->thresholds[curr_col + merge_start_col] = old_val;
    }
    i += count;
  }
}

// Merges elements in matches into a reduced distance matrix summary.
void Profile::matrix_merge(const std::vector<float> &values) {
  for (int i = 0; i < values.size(); ++i) {
    if (this->data[0].float_value[i] < values[i]) {
      this->data[0].float_value[i] = values[i];
    }
  }
}

void Profile::Alloc(size_t size, int64_t matrix_height, int64_t matrix_width,
                    float default_thresh) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      data.emplace_back();
      data[0].double_value.resize(size, 0);
      break;
    case PROFILE_TYPE_1NN:
      data.emplace_back();
      data[0].float_value.resize(size, -2.0);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = -2.0;
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
      thresholds.resize(size, default_thresh);
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      data.emplace_back();
      data[0].float_value.resize(matrix_height * matrix_width, -2.0);
      break;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      break;
  }
}

// Copies a profile to the host
void Profile::CopyFromDevice(const OpInfo *info, const ExecInfo *exec_info,
                             const DeviceProfile *device_tile_profile,
                             uint64_t length) {
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memcopy(this->data[0].double_value.data(),
              device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
              length * sizeof(double), true, exec_info);
      break;
    case PROFILE_TYPE_1NN:
      Memcopy(this->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN), length * sizeof(float),
              true, exec_info);
      break;
    case PROFILE_TYPE_1NN_INDEX:
      Memcopy(this->data[0].uint64_value.data(),
              device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
              length * sizeof(uint64_t), true, exec_info);
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      this->data[0].match_value_unordered.resize(length);
      Memcopy(this->data[0].match_value_unordered.data(),
              device_tile_profile->at(PROFILE_TYPE_APPROX_ALL_NEIGHBORS),
              length * sizeof(SCAMPmatch), true, exec_info);
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      Memcopy(this->data[0].float_value.data(),
              device_tile_profile->at(PROFILE_TYPE_MATRIX_SUMMARY),
              info->matrix_width * info->matrix_height * sizeof(float), true,
              exec_info);
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      ASSERT(false, "FUNCTIONALITY UNIMPLEMENTED");
      break;
  }
}

// Merges a profile corresponding to the result of a tile into this
// profile
void Profile::MergeTileToProfile(Profile *tile_profile, const OpInfo *info,
                                 uint64_t position, uint64_t length,
                                 uint64_t index_start) {
  // Check if we overflowed
  bool overflowed = length >= info->max_matches_per_tile;

  // Lock the before we merge, this function can be called by multiple
  // threads
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
      if (overflowed) {
        threshold_merge(tile_profile->data[0].match_value_unordered, position,
                        info->max_matches_per_column);
      } else {
        match_merge(tile_profile->data[0].match_value_unordered, index_start,
                    position, info->max_matches_per_column);
      }
      return;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      matrix_merge(tile_profile->data[0].float_value);
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
        "Error: Tile length and width must be at least 2x larger than "
        "the "
        "window size");
  }
  if (timeseries_a.size() < window || (has_b && timeseries_b.size() < window)) {
    throw SCAMPException(
        "Error: Input time series must be at least 'subesequence window "
        "size' "
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
    case PROFILE_TYPE_MATRIX_SUMMARY:
      return "PROFILE_TYPE_MATRIX_SUMMARY";
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
    case PRECISION_ULTRA:
      return "PRECISION_ULTRA";
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
    case PROFILE_TYPE_MATRIX_SUMMARY:
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
