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
    // Load the df/dg/inorm column+row data into smem (needed by the compute)
    // but skip the per-column/row profile init: matrix summary has no such
    // profile -- it uses the cell grid below. Passing COMPUTE_ROWS=COMPUTE_COLS
    // =false keeps the unconditional df/dg/inorm loads and drops the profile
    // writes.
    init_smem_with_static_initializer<SMEM_TYPE, PROFILE_DATA_TYPE,
                                      /*COMPUTE_ROWS=*/false,
                                      /*COMPUTE_COLS=*/false, tile_width,
                                      tile_height, BLOCKSZ>(
        args, smem, col_start, row_start, static_cast<PROFILE_DATA_TYPE>(0));
    // Orientation: the normal (upper) run has COMPUTE_COLS and writes profile_a
    // columnwise; the transposed (lower) run has COMPUTE_ROWS and writes
    // profile_b (which holds the real matrix in that run) with row/col swapped.
    // This mirrors the CPU update_columnwise vs update_rowwise split. Exactly
    // one flag is set per run for matrix summary.
    constexpr bool kRowwise = COMPUTE_ROWS;
    smem.ms_rowwise = kRowwise;
    smem.ms_matrix =
        reinterpret_cast<float *>(kRowwise ? profile_b : profile_a);
    // Size and zero the per-block cell grid for the cells this stripe-step
    // touches. The block walks a parallelogram (rows [row_start, +tile_height),
    // columns up to col_start + tile_width + tile_height); bucket both corners
    // with the run's orientation and take the spanning box. Bounds are computed
    // identically by every thread (no broadcast needed). The grid aliases the
    // local_mp_col region; if it needs more cells than fit, the grid is
    // disabled and the inner loop writes straight to global memory.
    int rb0, cb0, rb1, cb1;
    ms_cell_of(static_cast<double>(row_start), static_cast<double>(col_start),
               kRowwise, args, &rb0, &cb0);
    ms_cell_of(static_cast<double>(row_start + tile_height - 1),
               static_cast<double>(col_start + tile_width + tile_height - 1),
               kRowwise, args, &rb1, &cb1);
    smem.ms_row_min = rb0 < rb1 ? rb0 : rb1;
    smem.ms_col_min = cb0 < cb1 ? cb0 : cb1;
    smem.ms_grid_h = (rb0 > rb1 ? rb0 : rb1) - smem.ms_row_min + 1;
    smem.ms_grid_w = (cb0 > cb1 ? cb0 : cb1) - smem.ms_col_min + 1;
    smem.ms_num_cells = smem.ms_grid_w * smem.ms_grid_h;
    // Cap = capacity of the aliased region (local_mp_col on the normal run,
    // local_mp_row on the transposed run).
    constexpr int kMsGridCap =
        static_cast<int>(sizeof(PROFILE_DATA_TYPE) / sizeof(float)) *
        (kRowwise ? tile_height : tile_width);
    smem.ms_use_grid =
        smem.ms_num_cells > 0 && smem.ms_num_cells <= kMsGridCap;
    if (smem.ms_use_grid) {
      for (int idx = threadIdx.x; idx < smem.ms_num_cells; idx += BLOCKSZ) {
        smem.ms_grid[idx] = -2.0f;  // sentinel matches host matrix init
      }
    }
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
  if constexpr (PROFILE_TYPE == PROFILE_TYPE_MATRIX_SUMMARY) {
    // Flush the per-block smem cell grid to the global matrix with one atomic
    // per touched cell. Cells with no contribution keep the -2.0 sentinel and
    // are skipped. When the grid was disabled the inner loop already wrote
    // directly to global, so there is nothing to flush.
    if (smem.ms_use_grid) {
      for (int idx = threadIdx.x; idx < smem.ms_num_cells; idx += BLOCKSZ) {
        float v = smem.ms_grid[idx];
        if (v > -2.0f) {
          int lr = idx / smem.ms_grid_w;
          int lc = idx - lr * smem.ms_grid_w;
          int64_t gpos =
              static_cast<int64_t>(smem.ms_row_min + lr) * args.matrix_width +
              (smem.ms_col_min + lc);
          fAtomicMax<ATOMIC_GLOBAL>(smem.ms_matrix + gpos, v);
        }
      }
    }
    return;
  }
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
