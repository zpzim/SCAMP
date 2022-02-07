#include "common.h"
#include "scamp_exception.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace SCAMP {

static constexpr int64_t GIGABYTE = 1024 * 1024 * 1024;

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
      global_start_row_position(start_row),
      global_start_col_position(start_col),
      opt_args(args_),
      profile_type(profiletype),
      fp_type(t),
      self_join(selfjoin),
      computing_rows(compute_rows),
      computing_cols(compute_cols),
      is_aligned(aligned),
      keep_rows_separate(keep_rows),
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
    cols_per_cell =
        (full_ts_len_A - mp_window + 1) / static_cast<double>(matrix_width);
    rows_per_cell =
        (full_ts_len_B - mp_window + 1) / static_cast<double>(matrix_height);
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
  return "SCAMP_UNKNOWN_ERROR";
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
      throw SCAMPException("Error: Profile Type Unknown");
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
  return "PROFILE_TYPE_UNKNOWN";
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
  return "PRECISION_UNKNOWN";
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
