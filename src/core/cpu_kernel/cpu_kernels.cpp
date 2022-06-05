#include "cpu_kernels.h"

#include "core/defines.h"
#include "core/kernel_common.h"

#pragma STDC FP_CONTRACT ON

// When building a redistributable binary with runtime dispatch of AVX
// we need to make sure that each version of the binary we generate is
// ABI compatible with one another. Eigen detects the compile time availability
// of AVX and sets different alignments on memory allocations based on this.
#ifdef _SCAMP_DISTRIBUTABLE_
#define EIGEN_MAX_ALIGN_BYTES 32
#endif

#include <Eigen/Core>

#if defined(_SCAMP_USE_AVX_)
#define DISPATCHED_NAMESPACE AVX
#define UNROLL_WIDTH 256
#elif defined(_SCAMP_USE_AVX2_)
#define DISPATCHED_NAMESPACE AVX2
#define UNROLL_WIDTH 256
#elif defined(_SCAMP_DISTRIBUTABLE_)
#define DISPATCHED_NAMESPACE BASELINE
#define UNROLL_WIDTH 128
#else
// Non resdistributable binary with march=native
#define DISPATCHED_NAMESPACE BASELINE
#define UNROLL_WIDTH 256
#endif

namespace SCAMP {

// This file is compiled multple times with various compiler options and linked
// into a single binary. Namespaces are used to verify we are compiling all
// symbols in this file in each configuration. unrollWid is the amount of
// unrolling on the fast path.
namespace DISPATCHED_NAMESPACE {

// unrollWid is the amount of unrolling on the fast path.
constexpr int unrollWid{UNROLL_WIDTH};

struct ThreadInfo {
  ThreadInfo(const SCAMPKernelInputArgs<double> &args);
  int num_diags;
  int row_iters;
  int full_row_iters;
  int tile_diag;
  int row;
  int col;
};

ThreadInfo::ThreadInfo(const SCAMPKernelInputArgs<double> &args) {
  num_diags = args.n_x - args.exclusion_upper + 1;
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
    // There is a good amount of optimization possible here. Reusing computation
    // across calls to update_mp will allow for good speedup. However, floating
    // point roundoff error is a danger here, potentially causing output to the
    // wrong cell in the matrix, so we need to be careful to handle that
    // properly.
    int matrix_index;
    if constexpr (rowwise) {
      int col_idx =
          std::floor((info.row + args.global_start_col) / args.cols_per_cell);
      int row_idx =
          std::floor((info.col + args.global_start_row) / args.rows_per_cell);
      matrix_index = row_idx * args.matrix_width + col_idx;
    } else {
      int col_idx =
          std::floor((info.col + args.global_start_col) / args.cols_per_cell);
      int row_idx =
          std::floor((info.row + args.global_start_row) / args.rows_per_cell);
      matrix_index = row_idx * args.matrix_width + col_idx;
    }
    mp[matrix_index] = corr < args.opt.threshold || mp[matrix_index] >= corr
                           ? mp[matrix_index]
                           : corr;
  } else {
    ASSERT(false, "No Implementation provided for updating MP in CPU KERNEL");
  }
}

template <typename EIGEN_TYPE, SCAMPProfileType type>
FORCE_INLINE inline void reduce_row_fast(EIGEN_TYPE &corr, int &index,
                                         double thresh) {  // NOLINT
  switch (type) {
    case PROFILE_TYPE_1NN_INDEX: {
#if defined(_WIN32) || defined(__INTEL_COMPILER) || \
    defined(__INTEL_LLVM_COMPILER)
      // cl, clang-cl, and intel compilers work better using Eigen's reduction.
      corr[0] = corr.maxCoeff(&index);
#else
      // gcc and clang work better using a manual reduction loop.
      Eigen::Array<int, unrollWid, 1> corrIdx;
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
      index = corrIdx[0];
#endif
      break;
    }
    case PROFILE_TYPE_1NN: {
      // Eigen's reduction is faster in all cases here.
      corr[0] = corr.maxCoeff();
      break;
    }
    case PROFILE_TYPE_SUM_THRESH: {
      // TODO(zpzim): Provide a faster reduction implementation.
      corr[0] = (corr > thresh).select(corr, 0).sum();
      break;
    }
    default:
      break;
  }
}

// These reductions are slightly slower than what we can do with loops.
// If Eigen performance improves over time, these might end up faster.
template <typename EIGEN_TYPE, SCAMPProfileType type>
FORCE_INLINE inline void reduce_row_eigen(EIGEN_TYPE &corr, int &index,
                                          double thresh) {  // NOLINT
  switch (type) {
    case PROFILE_TYPE_1NN_INDEX: {
      corr[0] = corr.maxCoeff(&index);
      break;
    }
    case PROFILE_TYPE_1NN: {
      corr[0] = corr.maxCoeff();
      break;
    }
    case PROFILE_TYPE_SUM_THRESH: {
      corr[0] = (corr > thresh).select(corr, 0).sum();
      break;
    }
    default:
      break;
  }
}

template <typename EIGEN_TYPE, SCAMPProfileType type>
FORCE_INLINE inline void reduce_row(EIGEN_TYPE &corr, int &index,
                                    double thresh) {  // NOLINT

  if constexpr (EIGEN_TYPE::RowsAtCompileTime != Eigen::Dynamic) {
    reduce_row_fast<EIGEN_TYPE, type>(corr, index, thresh);
  } else {
    reduce_row_eigen<EIGEN_TYPE, type>(corr, index, thresh);
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE, typename EIGEN_TYPE,
          SCAMPProfileType PROFILE_TYPE>
FORCE_INLINE inline void update_columnwise(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    EIGEN_TYPE &corr, PROFILE_DATA_TYPE *__restrict profile) {
  info.col = info.tile_diag + info.row;
  for (int local_diag = 0; local_diag < corr.size(); local_diag++, info.col++) {
    update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, false>(
        profile, corr[local_diag], info, args);
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE, typename EIGEN_TYPE,
          SCAMPProfileType PROFILE_TYPE>
FORCE_INLINE inline void update_rowwise(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    EIGEN_TYPE &corr, PROFILE_DATA_TYPE *__restrict profile) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    info.col = info.tile_diag + info.row;
    for (int local_diag = 0; local_diag < corr.size();
         local_diag++, info.col++) {
      update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, true>(
          profile, corr[local_diag], info, args);
    }
  } else {
    int index = 0;
    reduce_row<EIGEN_TYPE, PROFILE_TYPE>(corr, index, args.opt.threshold);
    info.col = index + info.tile_diag + info.row;
    update_mp<DIST_TYPE, PROFILE_TYPE, PROFILE_DATA_TYPE, true>(
        profile, corr[0], info, args);
  }
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          typename EIGEN_CORR_TYPE, typename EIGEN_INPUT_TYPE,
          typename EIGEN_INPUT_TYPE_CONST, SCAMPProfileType PROFILE_TYPE,
          bool computing_rows, bool computing_cols>
FORCE_INLINE inline void handle_row(const SCAMPKernelInputArgs<double> &args,
                                    ThreadInfo &info, EIGEN_CORR_TYPE &corr,
                                    EIGEN_INPUT_TYPE &cov,
                                    EIGEN_INPUT_TYPE_CONST &normsa,
                                    EIGEN_INPUT_TYPE_CONST &dfa,
                                    EIGEN_INPUT_TYPE_CONST &dga,
                                    PROFILE_DATA_TYPE *__restrict profile_A,
                                    PROFILE_DATA_TYPE *__restrict profile_B) {
  corr = (cov * normsa * args.normsb[info.row]).template cast<DIST_TYPE>();
  if (args.has_nan_input) {
    // Remove any nan values so that they don't pollute the reduction.
    // This is expensive on some compilers so only do it if we need to.
    for (int local_diag = 0; local_diag < corr.size(); local_diag++) {
      corr[local_diag] = std::isfinite(corr[local_diag])
                             ? corr[local_diag]
                             : init_dist<DIST_TYPE, PROFILE_TYPE>();
    }
  }
  if constexpr (computing_cols) {
    update_columnwise<DIST_TYPE, PROFILE_DATA_TYPE, EIGEN_CORR_TYPE,
                      PROFILE_TYPE>(args, info, corr, profile_A);
  }
  if constexpr (computing_rows) {
    update_rowwise<DIST_TYPE, PROFILE_DATA_TYPE, EIGEN_CORR_TYPE, PROFILE_TYPE>(
        args, info, corr, profile_B);
  }
  cov += dfa * args.dgb[info.row];
  cov += dga * args.dfb[info.row];
}

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, bool computing_rows,
          bool computing_cols>
FORCE_INLINE inline void handle_row_fast(
    const SCAMPKernelInputArgs<double> &args, ThreadInfo &info,
    PROFILE_DATA_TYPE *__restrict profile_A,
    PROFILE_DATA_TYPE *__restrict profile_B) {
  Eigen::Array<DIST_TYPE, unrollWid, 1> corr;
  Eigen::Map<Eigen::Array<double, unrollWid, 1>> cov(args.cov + info.tile_diag);
  Eigen::Map<const Eigen::Array<double, unrollWid, 1>> normsa(
      args.normsa + info.tile_diag + info.row);
  Eigen::Map<const Eigen::Array<double, unrollWid, 1>> dfa(
      args.dfa + info.tile_diag + info.row);
  Eigen::Map<const Eigen::Array<double, unrollWid, 1>> dga(
      args.dga + info.tile_diag + info.row);
  handle_row<DIST_TYPE, PROFILE_DATA_TYPE, decltype(corr), decltype(cov),
             decltype(normsa), PROFILE_TYPE, computing_rows, computing_cols>(
      args, info, corr, cov, normsa, dfa, dga, profile_A, profile_B);
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
  Eigen::Array<DIST_TYPE, Eigen::Dynamic, 1> corr(diagmax - info.tile_diag);
  Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>> cov(
      args.cov + info.tile_diag, diagmax - info.tile_diag);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> normsa(
      args.normsa + info.tile_diag + info.row, diagmax - info.tile_diag);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dfa(
      args.dfa + info.tile_diag + info.row, diagmax - info.tile_diag);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dga(
      args.dga + info.tile_diag + info.row, diagmax - info.tile_diag);
  handle_row<DIST_TYPE, PROFILE_DATA_TYPE, decltype(corr), decltype(cov),
             decltype(normsa), PROFILE_TYPE, computing_rows, computing_cols>(
      args, info, corr, cov, normsa, dfa, dga, profile_A, profile_B);
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

    // Fast case where we can unroll fully.
    for (info.row = 0; info.row < info.full_row_iters; info.row++) {
      handle_row_fast<DIST_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE,
                      computing_rows, computing_cols>(args, info, profile_A,
                                                      profile_B);
    }

    // Slow case where we are too close to the edge to unroll fully.
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

}  // namespace DISPATCHED_NAMESPACE

}  // namespace SCAMP
