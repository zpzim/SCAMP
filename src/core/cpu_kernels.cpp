#include "cpu_kernels.h"
#include "defines.h"
#include "kernel_common.h"

#include <array>
#include <vector>

namespace SCAMP {

// this is hard coded for now. It needs to be a power of 2
// It may be desirable to make this larger, since the compiler is generally
// unable to fold everything into register names with either int or long long
// based indexing.
constexpr int unrollWid{256};

// set here for now. Could be set by cmake depending on what is supported. AVX
// and AVX2 benefit from at least 32 particularly if masked movement
// instructions are generated for the reduction steps.
constexpr int simdByteLen{32};

struct ThreadInfo {
  ThreadInfo(const SCAMPKernelInputArgs<double> &args);
  int num_diags;
  int row_iters;
  int full_row_iters;
  int tile_diag;
  int row;
  int col;
  double matrix_index;
  double col_increment;
  double row_increment;
  double row_position;
};

ThreadInfo::ThreadInfo(const SCAMPKernelInputArgs<double> &args) {
  num_diags = args.n_x - args.exclusion_upper + 1;
  col_increment = 1 / args.cols_per_cell;
  row_increment = 1 / args.rows_per_cell;
}

// Outputs an 'initial' distance value based on the type of profile being
// computed
template <typename DISTANCE_TYPE, SCAMPProfileType type>
FORCE_INLINE inline constexpr DISTANCE_TYPE init_dist() {
  switch (type) {
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_1NN:
    case PROFILE_TYPE_MATRIX_SUMMARY:
      // Smallest value possible is -1 so set to -2
      return static_cast<DISTANCE_TYPE>(-2);
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
    default:
      // We must set to 0 so we get an accurate sum
      return static_cast<DISTANCE_TYPE>(0);
  }
}

template <typename DIST_TYPE, SCAMPProfileType PROFILE_TYPE,
          typename PROFILE_DATA_TYPE, bool rowwise>
FORCE_INLINE inline void update_mp(PROFILE_DATA_TYPE *mp, double corr,
                                   ThreadInfo &info,
                                   const SCAMPKernelInputArgs<double> &args) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX) {
    int index = rowwise ? info.row : info.col;
    int match_index = rowwise ? info.col : info.row;
    if (corr > mp[index].floats[0]) {
      mp[index].floats[0] = corr;
      mp[index].ints[1] = match_index;
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    int index = rowwise ? info.row : info.col;
    mp[index] = mp[index] >= corr ? mp[index] : corr;
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    int index = rowwise ? info.row : info.col;
    mp[index] = corr > args.opt.threshold ? mp[index] + corr : mp[index];
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    int matrix_index = std::floor(info.matrix_index);
    mp[matrix_index] = corr < args.opt.threshold || mp[matrix_index] >= corr
                           ? mp[matrix_index]
                           : corr;
    if constexpr (rowwise) {
      info.row_position += info.row_increment;
      if (info.row_position >= 1) {
        info.row_position--;
        info.matrix_index += args.matrix_width;
      }
    } else {
      info.matrix_index += info.col_increment;
    }
  } else {
    ASSERT(false, "No Implementation provided for updating MP in CPU KERNEL");
  }
}

template <typename DATA_TYPE, SCAMPProfileType type>
FORCE_INLINE inline void reduce_row(std::array<DATA_TYPE, unrollWid> &corr,
                                    std::array<int, unrollWid / 2> &corrIdx,
                                    double thresh) {  // NOLINT
  switch (type) {
    case PROFILE_TYPE_MATRIX_SUMMARY:
    case PROFILE_TYPE_1NN_INDEX: {
      for (int i = 0; i < unrollWid / 2; i++) {
        corrIdx[i] = corr[i] >= corr[i + unrollWid / 2] ? i : i + unrollWid / 2;
        corr[i] = corr[i] >= corr[i + unrollWid / 2] ? corr[i]
                                                     : corr[i + unrollWid / 2];
      }
      auto horizontal_reduction = [&corr, &corrIdx](int offset) {
        for (int i = 0; i < offset; i++) {
          corrIdx[i] =
              corr[i] >= corr[i + offset] ? corrIdx[i] : corrIdx[i + offset];
          corr[i] = corr[i] >= corr[i + offset] ? corr[i] : corr[i + offset];
        }
      };
      for (int i = unrollWid / 4; i > 0; i /= 2) {
        horizontal_reduction(i);
      }
      break;
    }
    case PROFILE_TYPE_1NN: {
      auto horizontal_reduction = [&corr](int offset) {
        for (int i = 0; i < offset; i++) {
          corr[i] = corr[i] >= corr[i + offset] ? corr[i] : corr[i + offset];
        }
      };
      for (int i = unrollWid / 2; i > 0; i /= 2) {
        horizontal_reduction(i);
      }
      break;
    }
    case PROFILE_TYPE_SUM_THRESH: {
      DATA_TYPE sum = 0;
      for (int i = 0; i < unrollWid; i++) {
        if (corr[i] > thresh) {
          sum += corr[i];
        }
      }
      corr[0] = sum;
      break;
    }
    default:
      break;
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
FORCE_INLINE inline void update_columnwise(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    std::array<DIST_TYPE, unrollWid> &corr,
    PROFILE_DATA_TYPE *__restrict profile) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    double col_idx = (info.tile_diag + info.row + args.global_start_col) /
                     args.cols_per_cell;
    int row_idx =
        std::floor((info.row + args.global_start_row) / args.rows_per_cell);
    info.matrix_index = row_idx * args.matrix_width + col_idx;
  } else {
    info.col = info.tile_diag + info.row;
  }
  for (int local_diag = 0; local_diag < unrollWid; local_diag++, info.col++) {
    update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, false>(
        profile, corr[local_diag], info, args);
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
FORCE_INLINE inline void update_rowwise(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    std::array<DIST_TYPE, unrollWid> &corr,
    PROFILE_DATA_TYPE *__restrict profile) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    double col_idx = (info.row + args.global_start_col) / args.cols_per_cell;
    int row_idx =
        std::floor((info.tile_diag + info.row + args.global_start_row) /
                   args.rows_per_cell);
    info.matrix_index = row_idx * args.matrix_width + col_idx;
    info.row_position = ((info.tile_diag + info.row + args.global_start_row) /
                         args.rows_per_cell) -
                        row_idx;
    for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
      update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, true>(
          profile, corr[local_diag], info, args);
    }
  } else {
    std::array<int, unrollWid / 2>
        corrIdx;  // NOLINT(cppcoreguidelines-pro-type-member-init,
                  // hicpp-member-init)
    reduce_row<DIST_TYPE, PROFILE_TYPE>(corr, corrIdx, args.opt.threshold);
    info.col = corrIdx[0] + info.tile_diag + info.row;
    update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, true>(
        profile, corr[0], info, args);
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, bool computing_rows,
          bool computing_cols>
FORCE_INLINE inline void handle_row_fast(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    PROFILE_DATA_TYPE *__restrict profile_A,
    PROFILE_DATA_TYPE *__restrict profile_B) {
  alignas(simdByteLen) std::array<DIST_TYPE, unrollWid>
      corr;  // NOLINT(cppcoreguidelines-pro-type-member-init,
             // hicpp-member-init)
  // MSVC and other less sophisticated compilers cannot autovectorize this
  // loop unless all accesses appear aligned. We can trick these compilers
  // into vectorizing this loop by avoiding the complicated indexing
  // within the loop and rather just change the offest of each input array.
  double *__restrict cov = args.cov + info.tile_diag;
  const double *__restrict normsa = args.normsa + info.tile_diag + info.row;
  const double *__restrict normsb = args.normsb;
  for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
    corr[local_diag] =
        cov[local_diag] * normsa[local_diag] * normsb[info.row];
  }
  if (args.has_nan_input) {
    // Remove any nan values so that they don't pollute the reduction.
    // This is expensive on some compilers so only do it if we need to.
    for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
      corr[local_diag] = std::isfinite(corr[local_diag])
                             ? corr[local_diag]
                             : init_dist<DIST_TYPE, PROFILE_TYPE>();
    }
  }
  if constexpr (computing_cols) {
    update_columnwise<DIST_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE>(
        args, info, corr, profile_A);
  }
  if constexpr (computing_rows) {
    update_rowwise<DIST_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE>(args, info, corr,
                                                               profile_B);
  }
  // Same as above, avoid complicated indexing within the loop to get
  // less sophisticated compilers to vectorize it.
  const double *__restrict dfa = args.dfa + info.tile_diag + info.row;
  const double *__restrict dga = args.dga + info.tile_diag + info.row;
  double dfb = args.dfb[info.row];
  double dgb = args.dgb[info.row];
  for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
    cov[local_diag] += dfa[local_diag] * dgb;
    cov[local_diag] += dfb * dga[local_diag];
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, bool computing_rows,
          bool computing_cols>
FORCE_INLINE inline void handle_row_slow(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    PROFILE_DATA_TYPE *__restrict profile_A,
    PROFILE_DATA_TYPE *__restrict profile_B) {
  int diagmax = std::min(
      std::min(args.n_x - args.exclusion_upper + 1, args.n_x - info.row),
      info.tile_diag + unrollWid);
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    if constexpr (computing_cols) {
      double col_idx = (info.tile_diag + info.row + args.global_start_col) /
                       args.cols_per_cell;
      int row_idx =
          std::floor((info.row + args.global_start_row) / args.rows_per_cell);
      info.matrix_index = row_idx * args.matrix_width + col_idx;
    } else {
      double col_idx = (info.row + args.global_start_col) / args.cols_per_cell;
      int row_idx =
          std::floor((info.tile_diag + info.row + args.global_start_row) /
                     args.rows_per_cell);
      info.row_position = ((info.tile_diag + info.row + args.global_start_row) /
                           args.rows_per_cell) -
                          row_idx;
      info.matrix_index = row_idx * args.matrix_width + col_idx;
    }
  }
  for (int diag = info.tile_diag; diag < diagmax; diag++) {
    info.col = diag + info.row;
    DIST_TYPE corr =
        args.cov[diag] * args.normsa[info.col] * args.normsb[info.row];
    corr = std::isfinite(corr) ? corr : init_dist<DIST_TYPE, PROFILE_TYPE>();
    if constexpr (computing_cols) {
      update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, /*rowwise=*/false>(
          profile_A, corr, info, args);
    }
    if constexpr (computing_rows) {
      update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, /*rowwise=*/true>(
          profile_B, corr, info, args);
    }
  }
  for (int diag = info.tile_diag; diag < diagmax; diag++) {
    int col = diag + info.row;
    args.cov[diag] += args.dfa[col] * args.dgb[info.row];
    args.cov[diag] += args.dga[col] * args.dfb[info.row];
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, bool computing_rows,
          bool computing_cols>
void do_tile(const SCAMPKernelInputArgs<double> &args,
             PROFILE_DATA_TYPE *__restrict profile_A,
             PROFILE_DATA_TYPE *__restrict profile_B) {
  ThreadInfo info(args);

  for (info.tile_diag = args.exclusion_lower; info.tile_diag < info.num_diags;
       info.tile_diag += unrollWid) {
    // Determine the maximum number of iterations for this tile (includes slow
    // case)
    info.row_iters = std::min(args.n_x - info.tile_diag, args.n_y);

    // Determine how many optimized iterations we can do before the slow case
    if (info.tile_diag + unrollWid >= info.num_diags) {
      info.full_row_iters = 0;
    } else {
      info.full_row_iters = std::max(
          0, std::min(args.n_x - info.tile_diag - unrollWid + 1, args.n_y));
    }

    // Fast, Unrolled, Autovectorized Case
    for (info.row = 0; info.row < info.full_row_iters; info.row++) {
      handle_row_fast<DIST_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE,
                      computing_rows, computing_cols>(args, info, profile_A,
                                                      profile_B);
    }

    // Slow Case
    for (info.row = info.full_row_iters; info.row < info.row_iters;
         info.row++) {
      handle_row_slow<DIST_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE,
                      computing_rows, computing_cols>(args, info, profile_A,
                                                      profile_B);
    }
  }
}

template <typename DIST_TYPE, typename PROFILE_OUTPUT_TYPE,
          SCAMPProfileType PROFILE_TYPE>
SCAMPError_t LaunchDoTile(const SCAMPKernelInputArgs<double> &args,
                          PROFILE_OUTPUT_TYPE *profile_A,
                          PROFILE_OUTPUT_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          bool computing_cols) {
  if (computing_rows && computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  } else if (computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = false;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  } else if (computing_rows) {
    constexpr bool COMPUTE_COLS = false;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t compute_cpu_resources_and_launch(SCAMPKernelInputArgs<double> args,
                                              Tile *t, void *profile_a,
                                              void *profile_b, bool do_rows,
                                              bool do_cols) {
  int exclusion_total = args.exclusion_lower + args.exclusion_upper;
  if (exclusion_total < args.n_x) {
    switch (t->info()->profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double, PROFILE_TYPE_SUM_THRESH>(
            args, static_cast<double *>(profile_a),
            static_cast<double *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<float, mp_entry, PROFILE_TYPE_1NN_INDEX>(
            args, static_cast<mp_entry *>(profile_a),
            static_cast<mp_entry *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_1NN:
        return LaunchDoTile<float, float, PROFILE_TYPE_1NN>(
            args, static_cast<float *>(profile_a),
            static_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_MATRIX_SUMMARY:
        return LaunchDoTile<float, float, PROFILE_TYPE_MATRIX_SUMMARY>(
            args, static_cast<float *>(profile_a),
            static_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t cpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, false);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, false);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

SCAMPError_t cpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, false, true);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs<double> tile_args(t, true, true);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

}  // namespace SCAMP
