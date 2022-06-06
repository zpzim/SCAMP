#include "tile.h"
#include <algorithm>
#include <functional>
#ifdef _HAS_CUDA_
#include "kernels.h"
#endif
#include "cpu_kernel/kernel_dispatcher.h"

namespace SCAMP {

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
  int extra_exclusion = 0;
  if (info_->profile_type == PROFILE_TYPE_SUM_THRESH ||
      info_->profile_type == PROFILE_TYPE_FREQUENCY_THRESH) {
    // We need to omit the main diagonal from one tile so it doesn't get
    // double counted
    extra_exclusion = 1;
  }
  if (upper_tile) {
    exclusion = get_exclusion(info_->mp_window, get_tile_row(), get_tile_col());
    return std::make_pair(exclusion, 0);
  }
  size_t height = get_tile_height() - info_->mp_window + 1;
  exclusion =
      get_exclusion(info_->mp_window, get_tile_col(), get_tile_row() + height);
  return std::make_pair(extra_exclusion, exclusion);
}

// Gets the exclusion zone for a particular tile (logic for ab joins)
// AB-joins only need exclusion zones when they are part of a join where
// the inputs are aligned or part of a larger self-join.
// We use the member 'is_aligned' to represent this state.
std::pair<int, int> Tile::get_exclusion_for_ab_join(bool upper_tile) {
  int exclusion_lower = 0;
  int exclusion_upper = 0;

  int alternative_exclusion_lower = 0;
  // We need to omit the main diagonal from bordering subtiles in cases where
  // the join operator incorporates all values into the result to prevent
  // double counting of the main diagonal.
  if (upper_tile && (info_->profile_type == PROFILE_TYPE_SUM_THRESH ||
                     info_->profile_type == PROFILE_TYPE_FREQUENCY_THRESH)) {
    alternative_exclusion_lower += 1;
  }

  // If the global join is not 'aligned' most of this function is unnecessary
  if (!info_->is_aligned) {
    return std::make_pair(
        std::max(exclusion_lower, alternative_exclusion_lower),
        exclusion_upper);
  }

  size_t height = get_tile_height() - info_->mp_window + 1;
  size_t width = get_tile_width() - info_->mp_window + 1;
  int64_t start_col = get_tile_col();
  int64_t start_row = get_tile_row();

  // For distributed joins, we need to find the global position in the join
  // to determine the exclusion zone
  if (info_->global_start_col_position >= 0 &&
      info_->global_start_row_position >= 0) {
    start_col += info_->global_start_col_position;
    start_row += info_->global_start_row_position;
  }

  if (upper_tile) {
    // If we are an upper tile, we need to account for regions marked with a Y
    // in the diagram below:
    //            start_col
    //               |
    //               |
    //               V
    //
    // start_row --> Y X X X X Y Y <----top
    //                 Y X X X X Y
    //                   Y X X X X
    //                     Y X X X
    //                  ^    Y X X
    //                  |      Y X
    //               bottom      Y

    // On the main diagonal (bottom) we can have a trivial match in the case
    // that this AB join is part of a larger self-join.
    exclusion_lower = get_exclusion(info_->mp_window, start_row, start_col);

    // The top of the tile may need to be excluded if this tile is below
    // the main diagonal (start_row > start_col)
    if (start_row > start_col) {
      exclusion_upper =
          get_exclusion(info_->mp_window, start_row, start_col + width);
    } else {
      exclusion_upper = 0;
    }
  } else {
    // If we are a lower tile, we need to account for regions marked with a Y
    // in the diagram below:
    //           start_col
    //               |
    //               |
    //               V
    //
    // start_row --> Y
    //               X Y
    //               X X Y
    //               X X X Y <----top
    //               X X X X Y
    //               Y X X X X Y
    // bottom -----> Y Y X X X X Y
    // IMPORTANT NOTE: This tile is executed TRANSPOSED AS AN UPPER TILE
    // MEANING THE INPUTS INCLUDING THE EXCLUSION RANGES ARE REVERSED!!!

    // On the main diagonal (top) we can have a trivial match in the case
    // that this AB join is part of a larger self-join.
    exclusion_lower = get_exclusion(info_->mp_window, start_col, start_row);

    // The bottom of the tile may need to be excluded if this tile is above
    // the main diagonal (start_row <= start_col)
    if (start_row <= start_col) {
      exclusion_upper =
          get_exclusion(info_->mp_window, start_col, start_row + height);
    } else {
      exclusion_upper = 0;
    }
  }
  return std::make_pair(std::max(exclusion_lower, alternative_exclusion_lower),
                        exclusion_upper);
}

// Allocator for tile memory which can reside on the host or cuda devices
template <typename T>
T *alloc_mem(size_t count, SCAMPArchitecture arch, int deviceid) {
  switch (arch) {
    case CUDA_GPU_WORKER: {
#ifdef _HAS_CUDA_
      gpuErrchk(cudaSetDevice(deviceid));
      size_t bytes = count * sizeof(T);
      T *ptr;
      gpuErrchk(cudaMalloc(&ptr, bytes));
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
  ASSERT(false, "Architecture not defined");
  return nullptr;
}

// Deleter for tile memory which can reside on the host or cuda devices
template <typename T>
void free_mem(T *ptr, SCAMPArchitecture arch, int deviceid) {
  switch (arch) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      gpuErrchk(cudaSetDevice(deviceid));
      gpuErrchk(cudaFree(ptr));
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
  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      gpuErrchk(cudaSetDevice(get_cuda_id()));
      gpuErrchk(cudaMemsetAsync(destination, value, bytes, get_stream()));
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
  SCAMP::Memcopy(destination, source, bytes, from_tile, &exec_info_);
}

Tile::Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id)
    :  // Allocate memory for tile based on architecture
      T_A_dev_(alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id),
               // Lambda deallocator
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      T_B_dev_(alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      QT_dev_(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
              [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      means_A_(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      means_B_(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      norms_A_(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      norms_B_(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
               [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      df_A_(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      df_B_(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      dg_A_(alloc_mem<double>(info->max_tile_width, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      dg_B_(alloc_mem<double>(info->max_tile_height, arch, cuda_id),
            [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      scratchpad_(
          static_cast<double *>(
              alloc_mem<double>(info->max_tile_ts_size, arch, cuda_id)),
          [=](double *p) { return free_mem<double>(p, arch, cuda_id); }),
      thresholds_A_(
          alloc_mem<float>(info->max_tile_width, arch, cuda_id),
          [=](float *p) { return free_mem<float>(p, arch, cuda_id); }),
      thresholds_B_(
          alloc_mem<float>(info->max_tile_height, arch, cuda_id),
          [=](float *p) { return free_mem<float>(p, arch, cuda_id); }),
      profile_a_tile_(info->profile_type, info->max_tile_width,
                      info->opt_args.threshold, info->matrix_width,
                      info->matrix_height),
      profile_b_tile_(info->profile_type, info->max_tile_width,
                      info->opt_args.threshold, info->matrix_width,
                      info->matrix_height),
      scratch_(
          std::unique_ptr<qt_compute_helper>(new qt_compute_helper(  // NOLINT
              info->max_tile_ts_size, info->mp_window, true, arch))),
      current_tile_width_(0),
      current_tile_height_(0),
      current_tile_col_(0),
      current_tile_row_(0),
      info_(info),
      exec_info_(arch, cuda_id)

{
  size_t profile_size = GetProfileTypeSize(info_->profile_type);
  size_t rows_to_alloc, cols_to_alloc;

  // For profile types where we can have more than one match per tile we need
  // to allocate additional memory
  if (info_->profile_type == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    cols_to_alloc = info_->max_matches_per_tile;
    rows_to_alloc = info_->max_matches_per_tile;
  } else if (info_->profile_type == PROFILE_TYPE_MATRIX_SUMMARY) {
    cols_to_alloc = info_->matrix_height * info_->matrix_width;
    rows_to_alloc = 0;
  } else {
    cols_to_alloc = info_->max_tile_width;
    rows_to_alloc = info_->max_tile_height;
  }

  // Allocate the tile's device memory
  profile_a_tile_dev_[info_->profile_type] =
      alloc_mem<char>(profile_size * cols_to_alloc, arch, cuda_id);
  profile_b_tile_dev_[info_->profile_type] =
      alloc_mem<char>(profile_size * rows_to_alloc, arch, cuda_id);

  // Allocate variable to track number of outputs generated by the kernel
  profile_a_dev_length_ = alloc_mem<unsigned long long int>(1, arch, cuda_id);
  profile_b_dev_length_ = alloc_mem<unsigned long long int>(1, arch, cuda_id);
}

Tile::~Tile() {
  // Free any memory allocated that will not be freed automatically
  free_mem<char>(static_cast<char *>(profile_a_tile_dev_[info_->profile_type]),
                 get_arch(), get_cuda_id());
  free_mem<char>(static_cast<char *>(profile_b_tile_dev_[info_->profile_type]),
                 get_arch(), get_cuda_id());
  free_mem<unsigned long long int>(profile_a_dev_length_, get_arch(),
                                   get_cuda_id());
  free_mem<unsigned long long int>(profile_b_dev_length_, get_arch(),
                                   get_cuda_id());
}

void Tile::Sync() {
  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#if _HAS_CUDA_
      cudaStreamSynchronize(get_stream());
      gpuErrchk(cudaPeekAtLastError());
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
  Memcopy(T_A_dev_.get(), Ta_h.data() + current_tile_col_,
          sizeof(double) * current_tile_width_, false);
  Memcopy(T_B_dev_.get(), Tb_h.data() + current_tile_row_,
          sizeof(double) * current_tile_height_, false);
}

// Initializes the tile's local profile values based on global profiles
// "profile_a" and "profile_b"
SCAMPError_t Tile::InitProfile(Profile *profile_a, Profile *profile_b) {
  int profile_size = GetProfileTypeSize(info_->profile_type);
  int width = current_tile_width_ - info_->mp_window + 1;
  int height = current_tile_height_ - info_->mp_window + 1;
  SCAMPProfileType type = info_->profile_type;
  switch (type) {
    case PROFILE_TYPE_SUM_THRESH:
      Memset(profile_a_tile_dev_.at(type), 0, profile_size * width);
      Memset(profile_b_tile_dev_.at(type), 0, profile_size * height);
      break;
    case PROFILE_TYPE_1NN_INDEX: {
      const uint64_t *pA_ptr = profile_a->data[0].uint64_value.data();
      Memcopy(profile_a_tile_dev_.at(type), pA_ptr + current_tile_col_,
              sizeof(uint64_t) * width, false);
      if (info_->computing_rows && info_->keep_rows_separate) {
        const uint64_t *pB_ptr = profile_b->data[0].uint64_value.data();
        Memcopy(profile_b_tile_dev_.at(type), pB_ptr + current_tile_row_,
                sizeof(uint64_t) * height, false);
      } else if (info_->self_join) {
        Memcopy(profile_b_tile_dev_.at(type), pA_ptr + current_tile_row_,
                sizeof(uint64_t) * height, false);
      }
      break;
    }
    case PROFILE_TYPE_1NN: {
      const float *pA_ptr = profile_a->data[0].float_value.data();
      Memcopy(profile_a_tile_dev_.at(type), pA_ptr + current_tile_col_,
              sizeof(float) * width, false);
      if (info_->self_join) {
        Memcopy(profile_b_tile_dev_.at(type), pA_ptr + current_tile_row_,
                sizeof(float) * height, false);
      } else if (info_->computing_rows && info_->keep_rows_separate) {
        const float *pB_ptr = profile_b->data[0].float_value.data();
        Memcopy(profile_b_tile_dev_.at(type), pB_ptr + current_tile_row_,
                sizeof(float) * height, false);
      }
      break;
    }
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      Memset(profile_a_dev_length_, 0, sizeof(unsigned long long int));
      Memset(profile_b_dev_length_, 0, sizeof(unsigned long long int));
      Memcopy(thresholds_A_.get(),
              profile_a->thresholds.data() + current_tile_col_,
              sizeof(float) * width, false);
      if (info_->self_join) {
        Memcopy(thresholds_B_.get(),
                profile_a->thresholds.data() + current_tile_row_,
                sizeof(float) * height, false);
      } else if (info_->computing_rows && info_->keep_rows_separate) {
        Memcopy(thresholds_B_.get(),
                profile_b->thresholds.data() + current_tile_row_,
                sizeof(float) * height, false);
      }
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY: {
      const float *pA_ptr = profile_a->data[0].float_value.data();
      Memcopy(profile_a_tile_dev_.at(type), pA_ptr,
              sizeof(float) * profile_a->data[0].float_value.size(), false);
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

void Tile::InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b,
                     const CombinedStats &ab) {
  size_t bytes_a =
      (current_tile_width_ - info_->mp_window + 1) * sizeof(double);
  size_t bytes_b =
      (current_tile_height_ - info_->mp_window + 1) * sizeof(double);

  // If this tile contains nan inputs we will need to perform potentially more
  // expensive computation.
  has_nan_input_ = false;
  for (const auto &idx : a.nan_idxs()) {
    if (idx >= current_tile_col_ &&
        idx < current_tile_col_ + current_tile_width_) {
      has_nan_input_ = true;
      break;
    }
  }
  if (!has_nan_input_) {
    for (const auto &idx : b.nan_idxs()) {
      if (idx >= current_tile_row_ &&
          idx < current_tile_row_ + current_tile_height_) {
        has_nan_input_ = true;
        break;
      }
    }
  }

  // Initialize the tile's local stats based on global statistics "a" and "b"
  Memcopy(norms_A_.get(), a.norms().data() + current_tile_col_, bytes_a, false);
  Memcopy(norms_B_.get(), b.norms().data() + current_tile_row_, bytes_b, false);
  Memcopy(means_A_.get(), a.means().data() + current_tile_col_, bytes_a, false);
  Memcopy(means_B_.get(), b.means().data() + current_tile_row_, bytes_b, false);

  if (info_->fp_type == PRECISION_ULTRA) {
    // Initialize the tile's local stats when using alternative formula.
    Memcopy(df_A_.get(), ab.dc_bkwd.data() + current_tile_col_, bytes_a, false);
    Memcopy(dg_A_.get(), ab.dc_fwd.data() + current_tile_col_, bytes_a, false);
    Memcopy(df_B_.get(), ab.dr_fwd.data() + current_tile_row_, bytes_b, false);
    Memcopy(dg_B_.get(), ab.dr_bkwd.data() + current_tile_row_, bytes_b, false);
  } else {
    // Initialize the tile's local stats when using published formula.
    Memcopy(df_A_.get(), a.df().data() + current_tile_col_, bytes_a, false);
    Memcopy(dg_A_.get(), a.dg().data() + current_tile_col_, bytes_a, false);
    Memcopy(df_B_.get(), b.df().data() + current_tile_row_, bytes_b, false);
    Memcopy(dg_B_.get(), b.dg().data() + current_tile_row_, bytes_b, false);
  }
}

std::pair<int64_t, int64_t> Tile::get_profile_dims_from_device() {
  std::pair<int64_t, int64_t> result;
  result.first = 0;
  result.second = 0;
  this->Memcopy(&result.first, profile_a_dev_length_,
                sizeof(unsigned long long int), true);
  this->Memcopy(&result.second, profile_b_dev_length_,
                sizeof(unsigned long long int), true);
  Sync();
  if (result.first > info()->max_matches_per_tile) {
    if (!info_->silent_mode) {
      std::cout << "Warning: Unable to return all matches! SCAMP found a "
                   "total of "
                << result.first
                << " matches for this tile. But we could only store "
                << info_->max_matches_per_tile
                << " of them. Perhaps try a smaller tile size or a higher "
                   "match threshold? "
                << std::endl;
    }
    result.first = -1;
  }

  if (result.second > info()->max_matches_per_tile) {
    if (!info_->silent_mode) {
      std::cout << "Warning: Unable to return all matches! SCAMP found a "
                   "total of "
                << result.second
                << " matches for this tile. But we could only store "
                << info_->max_matches_per_tile
                << " of them. Perhaps try a smaller tile size or a higher "
                   "match threshold? "
                << std::endl;
    }
    result.second = -1;
  }
  if (!info_->silent_mode) {
    std::cout << "width = " << result.first << " height = " << result.second
              << std::endl;
  }
  return result;
}

void Tile::SortMatches(SCAMPmatch *matches, uint64_t len) {
#ifdef _HAS_CUDA_
  if (exec_info_.arch == CUDA_GPU_WORKER) {
    return match_gpu_sort(matches, len, get_stream());
  }
#endif
  ASSERT(false, "Sorting ALL_NEIGHBORS profiles is supported only on GPUs");
}

// TODO(zpzim): move this back into SCAMP_Operation, we shouldn't have the
// merging be functionality of the individual tile
bool Tile::MergeProfile(Profile *profile_a, Profile *profile_b) {
  // Set up a copy operation back to the host
  int height, width;
  switch (info_->profile_type) {
    case PROFILE_TYPE_1NN:
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_SUM_THRESH:
      // We already know how many elements to copy back
      width = current_tile_width_ - info_->mp_window + 1;
      height = current_tile_height_ - info_->mp_window + 1;
      break;
    case PROFILE_TYPE_MATRIX_SUMMARY:
      width = info_->matrix_width * info_->matrix_height;
      height = 0;
      break;
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS: {
      // We need to find the number of elements generated by the kernel
      auto width_height = get_profile_dims_from_device();
      width = width_height.first;
      height = width_height.second;
      break;
    }
    default:
      throw(SCAMPException("Functionality Unimplemented."));
      break;
  }

  bool overflowed = width < 0 || height < 0;

  if (width < 0) {
    width = info_->max_matches_per_tile;
  }
  if (height < 0) {
    height = info_->max_matches_per_tile;
  }

  if (NeedsSort(info_->profile_type)) {
    // Sort the resulting array
    SortMatches(
        static_cast<SCAMPmatch *>(profile_a_tile_dev_.at(info_->profile_type)),
        width);
    if (info_->computing_rows) {
      SortMatches(static_cast<SCAMPmatch *>(
                      profile_b_tile_dev_.at(info_->profile_type)),
                  height);
    }
  }

  profile_a_tile_.CopyFromDevice(info_, &exec_info_, &profile_a_tile_dev_,
                                 width);
  if (info_->computing_rows) {
    profile_b_tile_.CopyFromDevice(info_, &exec_info_, &profile_b_tile_dev_,
                                   height);
  }

  // Wait for the previous work to be done
  Sync();

  // Merge result
  profile_a->MergeTileToProfile(&profile_a_tile_, info_, current_tile_col_,
                                width, current_tile_row_, overflowed);

  if (info_->computing_rows && info_->keep_rows_separate) {
    profile_b->MergeTileToProfile(&profile_b_tile_, info_, current_tile_row_,
                                  height, current_tile_col_, overflowed);
  } else if (info_->self_join && info_->computing_rows) {
    profile_a->MergeTileToProfile(&profile_b_tile_, info_, current_tile_row_,
                                  height, current_tile_col_, overflowed);
  }

  return !overflowed;
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

  // Compute the upper triangular portion of the tile
  error = do_self_join_half();
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  // Compute the lower triangular portion of the tile based on worker arch
  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          scratch_->compute_QT(QT_dev_.get(), T_B_dev_.get(), T_A_dev_.get(),
                               means_A_.get(), get_stream());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_lower(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = scratch_->compute_QT_CPU(QT_dev_.get(), T_B_dev_.get(),
                                       T_A_dev_.get());
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

  if (info_->mp_window > current_tile_width_) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (info_->mp_window > current_tile_height_) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  // Compute the upper triangular portion of the tile based on worker arch
  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          scratch_->compute_QT(QT_dev_.get(), T_A_dev_.get(), T_B_dev_.get(),
                               means_B_.get(), get_stream());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_self_join_upper(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = scratch_->compute_QT_CPU(QT_dev_.get(), T_A_dev_.get(),
                                       T_B_dev_.get());
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

  if (info_->mp_window > current_tile_width_) {
    return SCAMP_DIM_INCOMPATIBLE;
  }
  if (info_->mp_window > current_tile_height_) {
    return SCAMP_DIM_INCOMPATIBLE;
  }

  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          scratch_->compute_QT(QT_dev_.get(), T_A_dev_.get(), T_B_dev_.get(),
                               means_B_.get(), get_stream());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = gpu_kernel_ab_join_upper(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = scratch_->compute_QT_CPU(QT_dev_.get(), T_A_dev_.get(),
                                       T_B_dev_.get());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_ab_join_upper(this);
      break;
  }
  if (error != SCAMP_NO_ERROR) {
    return error;
  }

  switch (get_arch()) {
    case CUDA_GPU_WORKER:
#ifdef _HAS_CUDA_
      error =
          scratch_->compute_QT(QT_dev_.get(), T_B_dev_.get(), T_A_dev_.get(),
                               means_A_.get(), get_stream());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }

      error = gpu_kernel_ab_join_lower(this);
#else
      ASSERT(false, "ERROR: CUDA used in binary not built with CUDA");
#endif
      break;
    case CPU_WORKER:
      error = scratch_->compute_QT_CPU(QT_dev_.get(), T_B_dev_.get(),
                                       T_A_dev_.get());
      if (error != SCAMP_NO_ERROR) {
        return error;
      }
      error = cpu_kernel_ab_join_lower(this);
      break;
  }
  return error;
}

}  // namespace SCAMP
