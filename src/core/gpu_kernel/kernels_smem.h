#pragma once

/////////////////////////////////////////////////////
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
// init_smem dispatches by profile type to one of three init strategies:
//   - dynamic_initializer: profile values are seeded from the global
//     matrix profile passed in (used by 1NN / 1NN_INDEX, where any prior
//     best-so-far for this tile's columns/rows is the right starting
//     point).
//   - static_initializer: profile values are seeded from a constant
//     (used by SUM_THRESH where the running sum starts at 0, and by
//     MATRIX_SUMMARY where the threshold is baked into a single
//     mp_entry sentinel).
//   - all_neighbors_initializer: per-position threshold seeding for
//     APPROX_ALL_NEIGHBORS.
//
// The smem parameter type is templated (`SMEM_TYPE`) rather than spelled
// SCAMPSmem<...> so callers don't have to thread tile_width/tile_height
// through every signature.
//////////////////////////////////////////////////

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_with_static_initializer(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem, uint32_t col_start,
    uint32_t row_start, PROFILE_DATA_TYPE initializer) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_with_dynamic_initializer(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem,
    PROFILE_DATA_TYPE *initializer_col, PROFILE_DATA_TYPE *initializer_row,
    uint32_t col_start, uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer_col[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer_row[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

// Per-position threshold-seeded init for APPROX_ALL_NEIGHBORS.
template <typename SMEM_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ inline void init_smem_for_all_neighbors(
    SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem, uint32_t col_start,
    uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  mp_entry initializer;
  initializer.ints[1] = 0;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      initializer.floats[0] = args.thresholds_a[global_position];
      smem.local_mp_col[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      initializer.floats[0] = args.thresholds_b[global_position];
      smem.local_mp_row[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename SMEM_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
__device__ void init_smem(SCAMPKernelInputArgs<double> &args, SMEM_TYPE &smem,
                          PROFILE_OUTPUT_TYPE *profile_a,
                          PROFILE_OUTPUT_TYPE *profile_b, uint32_t col_start,
                          uint32_t row_start) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                PROFILE_TYPE == PROFILE_TYPE_1NN) {
    init_smem_with_dynamic_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                       COMPUTE_ROWS, COMPUTE_COLS, tile_width,
                                       tile_height, BLOCKSZ>(
        args, smem, profile_a, profile_b, col_start, row_start);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    init_smem_with_static_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                      COMPUTE_ROWS, COMPUTE_COLS, tile_width,
                                      tile_height, BLOCKSZ>(
        args, smem, col_start, row_start, 0.0);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    mp_entry e;
    e.floats[0] = args.opt.threshold;
    e.ints[1] = 0;
    init_smem_with_static_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                      COMPUTE_ROWS, COMPUTE_COLS, tile_width,
                                      tile_height, BLOCKSZ>(
        args, smem, col_start, row_start, e.ulong);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    init_smem_for_all_neighbors<SMEM_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                                tile_width, tile_height, BLOCKSZ>(
        args, smem, col_start, row_start);
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "init_smem not implemented for profile type.");
  }
}

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

template <SCAMPProfileType PROFILE_TYPE, typename DerivedProfile,
          typename DerivedSmem>
__device__ inline void write_back_value(
    SCAMPKernelInputArgs<double> &args, int local_position, int global_position,
    const Eigen::ArrayBase<DerivedSmem> &smem_profile, DerivedProfile *profile,
    unsigned long long int *profile_length, const float *thresholds) {
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    fAtomicMax<ATOMIC_GLOBAL>(profile + global_position,
                              smem_profile[local_position]);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX) {
    mp_entry e;
    e.ulong = smem_profile[local_position];
    MPatomicMax<ATOMIC_GLOBAL>(profile + global_position, e.floats[0],
                               e.ints[1]);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    do_atomicAdd<DerivedProfile, ATOMIC_GLOBAL>(profile + global_position,
                                                smem_profile[local_position]);
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Both COMPUTE_COLS and COMPUTE_ROWS paths land here; for matrix summary
    // the per-cell aggregation uses the "global_position is column" identity
    // (the row branch only runs in the transposed configuration).
    mp_entry e;
    e.ulong = smem_profile[local_position];
    if (e.floats[0] > args.opt.threshold) {
      int col = (global_position + args.global_start_col) / args.cols_per_cell;
      int row = (e.ints[1] + args.global_start_row) / args.rows_per_cell;
      fAtomicMax<ATOMIC_GLOBAL>(profile + (row * args.matrix_width + col),
                                e.floats[0]);
    }
  } else if constexpr (PROFILE_TYPE == PROFILE_TYPE_APPROX_ALL_NEIGHBORS) {
    mp_entry e;
    e.ulong = smem_profile[local_position];
    if (e.floats[0] > thresholds[global_position]) {
      unsigned long long int pos =
          do_atomicAdd<unsigned long long int, ATOMIC_GLOBAL>(profile_length,
                                                              1ULL);
      if (pos < args.max_matches_per_tile) {
        profile[pos].corr = e.floats[0];
        profile[pos].row = e.ints[1];
        profile[pos].col = global_position;
      }
    }
  } else {
    static_assert(PROFILE_TYPE != PROFILE_TYPE_INVALID,
                  "write_back_value not implemented for profile type.");
  }
}

template <SCAMPProfileType PROFILE_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int BLOCKSZ, int TILE_WIDTH, int TILE_HEIGHT, typename DerivedProfile,
          typename DerivedSmem>
__device__ void write_back(SCAMPKernelInputArgs<double> &args,
                           DerivedSmem &smem, uint32_t tile_start_x,
                           uint32_t tile_start_y, uint32_t n_x, uint32_t n_y,
                           DerivedProfile *profile_A,
                           DerivedProfile *profile_B) {
  int global_position, local_position;
  // The match-output atomic counter has to target *global* memory: that
  // counter is what Profile::CopyFromDevice reads back to size the
  // match_value_unordered vector for the host. smem.profile_a_length is the
  // smem-cached *copy* of that counter, used inside do_tile by the
  // NeedsCheckIfDone early-exit logic to test whether
  // max_matches_per_tile is exhausted -- if write_back atomicAdd-s into
  // the smem copy, the global counter stays at zero and the host sees an
  // empty profile. Regression introduced by the Eigen port (687a70b);
  // the pre-Eigen write_back used args.profile_{a,b}_length directly.
  if constexpr (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      write_back_value<PROFILE_TYPE>(args, local_position, global_position,
                                     smem.local_mp_col, profile_A,
                                     args.profile_a_length, args.thresholds_a);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if constexpr (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      write_back_value<PROFILE_TYPE>(args, local_position, global_position,
                                     smem.local_mp_row, profile_B,
                                     args.profile_b_length, args.thresholds_b);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}
