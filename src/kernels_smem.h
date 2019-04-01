#pragma once

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE, typename = void>
class InitMemStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(
      SCAMPKernelInputArgs<double> &args,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
      PROFILE_DATA_TYPE *__restrict__ profile_a,
      PROFILE_DATA_TYPE *__restrict__ profile_B, uint32_t col_start,
      uint32_t row_start) {
    assert(false);
  }

 protected:
  __device__ InitMemStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE,
                      std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH>>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ void exec(
      SCAMPKernelInputArgs<double> &args,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
      PROFILE_DATA_TYPE *__restrict__ profile_a,
      PROFILE_DATA_TYPE *__restrict__ profile_B, uint32_t col_start,
      uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.dg_col[local_position] = args.dga[global_position];
      smem.df_col[local_position] = args.dfa[global_position];
      smem.inorm_col[local_position] = args.normsa[global_position];
      if (COMPUTE_COLS) {
        smem.local_mp_col[local_position] = 0.0;
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
        smem.local_mp_row[local_position] = 0.0;
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE,
                      std::enable_if_t<PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX ||
                                       PROFILE_TYPE == PROFILE_TYPE_1NN>>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ virtual void exec(
      SCAMPKernelInputArgs<double> &args,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
      PROFILE_DATA_TYPE *__restrict__ profile_a,
      PROFILE_DATA_TYPE *__restrict__ profile_b, uint32_t col_start,
      uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.dg_col[local_position] = args.dga[global_position];
      smem.df_col[local_position] = args.dfa[global_position];
      smem.inorm_col[local_position] = args.normsa[global_position];
      if (COMPUTE_COLS) {
        smem.local_mp_col[local_position] = profile_a[global_position];
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
        smem.local_mp_row[local_position] = profile_b[global_position];
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

// Dummy (forces compilation failure when the wrong types are used)
template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class WriteBackStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    assert(false);
  }

 protected:
  __device__ WriteBackStrategy() {}
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
            profile_A + global_position, local_mp_col[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
            profile_B + global_position, local_mp_row[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE_1NN>
    : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        fAtomicMax<ATOMIC_GLOBAL>(profile_A + global_position,
                                  local_mp_col[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        fAtomicMax<ATOMIC_GLOBAL>(profile_B + global_position,
                                  local_mp_row[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        mp_entry e;
        e.ulong = local_mp_col[local_position];
        MPatomicMax<ATOMIC_GLOBAL>(profile_A + global_position, e.floats[0],
                                   e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        mp_entry e;
        e.ulong = local_mp_row[local_position];
        MPatomicMax<ATOMIC_GLOBAL>(profile_B + global_position, e.floats[0],
                                   e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};
