#pragma once

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (common, optimized case)
//
//
//////////////////////////////////////////////////

// Catch-all strategy, should never be used
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, typename = void>
class DoRowOptStrategy : SCAMPStrategy {
 public:
  __device__ void exec(ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3,
                       ACCUM_TYPE &cov4, DISTANCE_TYPE &distc1,
                       DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
                       DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx,
                       const DATA_TYPE &inormcy, const DATA_TYPE &inormcz,
                       const DATA_TYPE &inormcw, const DATA_TYPE &inormr,
                       const DATA_TYPE &df_colx, const DATA_TYPE &df_coly,
                       const DATA_TYPE &df_colz, const DATA_TYPE &df_colw,
                       const DATA_TYPE &dg_colx, const DATA_TYPE &dg_coly,
                       const DATA_TYPE &dg_colz, const DATA_TYPE &dg_colw,
                       const DATA_TYPE &df_row, const DATA_TYPE &dg_row,
                       const int &row, const int &col, const int &global_row,
                       const int &global_col,
                       PROFILE_DATA_TYPE *__restrict__ mp_row,
                       OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowOptStrategy() {}
};

// Sum-threshold join strategy
template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy<
    DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
    COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH>>
    : public SCAMPStrategy {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
      OptionalArgs &args) {
    DISTANCE_TYPE distx = info.cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = info.cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = info.cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = info.cov4 * inormcw * inormr;
    DISTANCE_TYPE thresh = args.threshold;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;

    DISTANCE_TYPE count_row = 0;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distx;
      }
      if (COMPUTE_COLS) {
        distc1 += distx;
      }
    }
    if (disty > thresh) {
      if (COMPUTE_ROWS) {
        count_row += disty;
      }
      if (COMPUTE_COLS) {
        distc2 += disty;
      }
    }
    if (distz > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distz;
      }
      if (COMPUTE_COLS) {
        distc3 += distz;
      }
    }
    if (distw > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distw;
      }
      if (COMPUTE_COLS) {
        distc4 += distw;
      }
    }
    // coalesce all row updates to lane 0 of each warp and atomically update
    // This way is more efficient than atomics when we expect a lot of updates
    if (COMPUTE_ROWS) {
#pragma unroll
      for (int i = 16; i >= 1; i /= 2) {
        count_row += __shfl_down_sync(0xffffffff, count_row, i);
      }
      if ((threadIdx.x & 0x1f) == 0) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
            smem.local_mp_row + info.local_row, count_row);
      }
    }
    info.local_row++;
    info.local_col++;
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy<
    DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
    COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX>> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      uint32_t &idxc1, uint32_t &idxc2, uint32_t &idxc3, uint32_t &idxc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, float &curr_mp_row_val,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, OptionalArgs &args) {
    DISTANCE_TYPE distx = info.cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = info.cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = info.cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = info.cov4 * inormcw * inormr;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;
    // Update the column best-so-far values

    if (COMPUTE_COLS) {
      MPMax2(distc1, distx, idxc1, info.global_row);
      MPMax2(distc2, disty, idxc2, info.global_row);
      MPMax2(distc3, distz, idxc3, info.global_row);
      MPMax2(distc4, distw, idxc4, info.global_row);
    }

    if (COMPUTE_ROWS) {
      // We take the maximum of the columns we computed for the row
      // And use that value to check the matrix profile
      uint32_t idx;
      DISTANCE_TYPE d =
          max4<DISTANCE_TYPE>(distx, disty, distz, distw, info.global_col, idx);
      MPatomicMax_check((uint64_t *)(smem.local_mp_row + info.local_row), d,
                        idx, curr_mp_row_val);
    }

    info.local_row++;
    info.local_col++;
    info.global_row++;
    info.global_col++;
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
                       std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN>> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, float &distc1, float &distc2,
      float &distc3, float &distc4, const DATA_TYPE &inormcx,
      const DATA_TYPE &inormcy, const DATA_TYPE &inormcz,
      const DATA_TYPE &inormcw, const DATA_TYPE &inormr,
      const DATA_TYPE &df_colx, const DATA_TYPE &df_coly,
      const DATA_TYPE &df_colz, const DATA_TYPE &df_colw,
      const DATA_TYPE &dg_colx, const DATA_TYPE &dg_coly,
      const DATA_TYPE &dg_colz, const DATA_TYPE &dg_colw,
      const DATA_TYPE &df_row, const DATA_TYPE &dg_row, float &curr_mp_row_val,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, OptionalArgs &args) {
    float distx = info.cov1 * inormcx * inormr;
    float disty = info.cov2 * inormcy * inormr;
    float distz = info.cov3 * inormcz * inormr;
    float distw = info.cov4 * inormcw * inormr;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;
    // Update the column best-so-far values

    if (COMPUTE_COLS) {
      distc1 = fmaxf(distc1, distx);
      distc2 = fmaxf(distc2, disty);
      distc3 = fmaxf(distc3, distz);
      distc4 = fmaxf(distc4, distw);
    }

    if (COMPUTE_ROWS) {
      // We take the maximum of the columns we computed for the row
      // And use that value to check the matrix profile
      float d = fmaxf(fmaxf(distx, disty), fmaxf(distz, distw));
      fAtomicMax_check<ATOMIC_BLOCK>(smem.local_mp_row + info.local_row, d,
                                     curr_mp_row_val);
    }

    info.local_row++;
    info.local_col++;
    info.global_row++;
    info.global_col++;
  }
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (uncommon, edge case)
//
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, typename = void>
class DoRowEdgeStrategy : SCAMPStrategy {
 public:
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowEdgeStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy<
    DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
    COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH>> : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    DISTANCE_TYPE distr = 0;
    DISTANCE_TYPE distx, disty, distz, distw;
    DISTANCE_TYPE thresh = static_cast<DISTANCE_TYPE>(args.threshold);
    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;
    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        distr += distx;
      }
      if (COMPUTE_COLS) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_col + j,
                                                      distx);
      }
    }
    if (x + 1 < n && diag + 1 < num_diags) {
      if (disty > thresh) {
        if (COMPUTE_ROWS) {
          distr += disty;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 1, disty);
        }
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (distz > thresh) {
        if (COMPUTE_ROWS) {
          distr += distz;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 2, distz);
        }
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (distw > thresh) {
        if (COMPUTE_ROWS) {
          distr += distw;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 3, distw);
        }
      }
    }
    if (COMPUTE_ROWS) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_row + i,
                                                    distr);
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy<
    DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
    COMPUTE_COLS, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX>> : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    float dist_row;
    uint32_t idx_row;
    float distx;
    float disty;
    float distz;
    float distw;

    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (COMPUTE_COLS) {
      MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_col + j), distx, y);
    }
    dist_row = distx;
    idx_row = x;
    if (x + 1 < n && diag + 1 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, disty, idx_row, x + 1, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_col + j + 1),
                                  disty, y);
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distz, idx_row, x + 2, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_col + j + 2),
                                  distz, y);
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distw, idx_row, x + 3, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_col + j + 3),
                                  distw, y);
      }
    }
    if (COMPUTE_ROWS) {
      MPatomicMax<ATOMIC_BLOCK>((uint64_t *)(smem.local_mp_row + i), dist_row,
                                idx_row);
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE,
                        std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN>>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    float dist_row;
    float distx;
    float disty;
    float distz;
    float distw;

    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (COMPUTE_COLS) {
      fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_col + j), distx);
    }
    dist_row = distx;
    if (x + 1 < n && diag + 1 < num_diags) {
      if (COMPUTE_ROWS) {
        dist_row = fmaxf(dist_row, disty);
      }
      if (COMPUTE_COLS) {
        fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_col + j + 1), disty);
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (COMPUTE_ROWS) {
        dist_row = fmaxf(dist_row, distz);
      }
      if (COMPUTE_COLS) {
        fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_col + j + 2), distz);
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (COMPUTE_ROWS) {
        dist_row = fmaxf(dist_row, distw);
      }
      if (COMPUTE_COLS) {
        fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_col + j + 3), distw);
      }
    }
    if (COMPUTE_ROWS) {
      fAtomicMax<ATOMIC_BLOCK>((float *)(smem.local_mp_row + i), dist_row);
    }
  }
};

//////////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR UPDATING THE COLUMNS OF THE LOCAL MP VALUES IN
// THE OPTIMIZED CASE
//
//////////////////////////////////////////////////////////////////////

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, typename = void>
class UpdateColumnsStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }

 protected:
  __device__ UpdateColumnsStrategy() {}
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class UpdateColumnsStrategy<
    DISTANCE_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX>>
    : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class UpdateColumnsStrategy<
    DISTANCE_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE,
    std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH>>
    : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      DISTANCE_TYPE distc1, DISTANCE_TYPE distc2, DISTANCE_TYPE distc3,
      DISTANCE_TYPE distc4, DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
      DISTANCE_TYPE distc7, PROFILE_DATA_TYPE *__restrict__ local_mp_col,
      uint64_t col) {
    int lane = threadIdx.x & 0x1f;
    DISTANCE_TYPE overlap_1, overlap_2, overlap_3;

    // Send the overlapping sums to the next thread
    overlap_1 = __shfl_up_sync(0xffffffff, distc5, 1);
    overlap_2 = __shfl_up_sync(0xffffffff, distc6, 1);
    overlap_3 = __shfl_up_sync(0xffffffff, distc7, 1);
    if (lane > 0) {
      distc1 += overlap_1;
      distc2 += overlap_2;
      distc3 += overlap_3;
    }
    // Update the shared memory sums
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col, distc1);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 1,
                                                  distc2);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 2,
                                                  distc3);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 3,
                                                  distc4);
    // The last thread in the warp has to make additional updates to shared
    // memory as it had nowhere to send its overlapping sums
    if (lane == 31) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 4,
                                                    distc5);
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 5,
                                                    distc6);
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 6,
                                                    distc7);
    }
  }
};
