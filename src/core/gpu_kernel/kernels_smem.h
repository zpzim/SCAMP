#pragma once

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
//
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

template <typename SMEM_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem_for_all_neighbors(SCAMPKernelInputArgs<double> &args,
                                            SMEM_TYPE &smem, uint32_t col_start,
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
          int tile_width, int tile_height, int BLOCKSZ, int PROFILE_TYPE>
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
    static_assert(PROFILE_TYPE != -1,
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
__device__ void write_back_value(
    SCAMPKernelInputArgs<double> &args, int local_position, int global_position,
    const Eigen::ArrayBase<DerivedSmem> &smem_profile, DerivedProfile *profile,
    uint64_t *profile_length, const float *thresholds) {
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
      // Reserve space in output array
      unsigned long long int pos =
          do_atomicAdd<unsigned long long int, ATOMIC_GLOBAL>(
              reinterpret_cast<unsigned long long int *>(profile_length), 1ULL);
      // Write the match to the output
      if (pos < args.max_matches_per_tile) {
        profile[pos].corr = e.floats[0];
        profile[pos].row = e.ints[1];
        profile[pos].col = global_position;
      }
    }
  } else {
    static_assert(PROFILE_TYPE != -1,
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
  if constexpr (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      write_back_value<PROFILE_TYPE>(args, local_position, global_position,
                                     smem.local_mp_col, profile_A,
                                     smem.profile_a_length, args.thresholds_a);
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
                                     smem.profile_b_length, args.thresholds_b);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}
