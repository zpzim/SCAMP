#include "cpu_kernels.h"
#include <vector>

namespace SCAMP {

template <bool computing_rows, bool computing_cols>
static inline void partialcross_kern(
    double* __restrict cov, double* __restrict mpa, int* __restrict mpia,
    double* __restrict mpb, int* __restrict mpib, const double* __restrict dfa,
    const double* __restrict dga, const double* __restrict invna,
    const double* __restrict dfb, const double* __restrict dgb,
    const double* __restrict invnb, const int amx, const int bmx,
    const int amin, const int upper_excl) {
  for (int ia = amin; ia < amx - upper_excl; ia++) {
    int mx = std::min(amx - ia, bmx);
    for (int ib = 0; ib < mx; ib++) {
      double cr = cov[ia] * invna[ib + ia] * invnb[ib];
      if (computing_cols) {
        if (cr > mpa[ib + ia]) {
          mpa[ib + ia] = cr;
          mpia[ib + ia] = ib;
        }
      }
      if (computing_rows) {
        if (cr > mpb[ib]) {
          mpb[ib] = cr;
          mpib[ib] = ib + ia;
        }
      }
      cov[ia] += dfa[ib + ia] * dgb[ib];
      cov[ia] += dfb[ib] * dga[ib + ia];
    }
  }
}

static int get_exclusion(uint64_t window_size, int64_t start_row,
                         int64_t start_column) {
  int exclusion = window_size / 4;
  if (start_column >= start_row && start_column <= start_row + exclusion) {
    return exclusion;
  }
  return 0;
}

static std::pair<int, int> get_exclusion_for_ab_join(uint64_t window_size,
                                                     uint64_t start_row,
                                                     uint64_t start_column,
                                                     bool upper_tile,
                                                     int tile_dim) {
  int exclusion_lower = 0;
  int exclusion_upper = 0;
  if (upper_tile) {
    exclusion_lower = get_exclusion(window_size, start_row, start_column);
    if (start_row > start_column) {
      exclusion_upper =
          get_exclusion(window_size, start_row, start_column + tile_dim);
    } else {
      exclusion_upper = 0;
    }
    return std::make_pair(exclusion_lower, exclusion_upper);
  }
  exclusion_lower = get_exclusion(window_size, start_column, start_row);
  if (start_row >= start_column) {
    exclusion_upper = 0;
  } else {
    exclusion_upper =
        get_exclusion(window_size, start_column, start_row + tile_dim);
  }
  return std::make_pair(exclusion_lower, exclusion_upper);
}

void split_profile(std::vector<double>& mp, std::vector<int32_t>& mpi,
                   uint64_t* profile, int len) {
  mp_entry e;
  for (int i = 0; i < len; ++i) {
    e.ulong = profile[i];
    mp[i] = static_cast<double>(e.floats[0]);
    mpi[i] = e.ints[1];
  }
}

void combine_profile(std::vector<double>& mp, std::vector<int32_t>& mpi,
                     uint64_t* profile, int len) {
  mp_entry e;
  for (int i = 0; i < len; ++i) {
    e.floats[0] = static_cast<float>(mp[i]);
    e.ints[1] = mpi[i];
    profile[i] = e.ulong;
  }
}

SCAMPError_t cpu_kernel_self_join_upper(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);
  split_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()), width);
  split_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()), height);
  const int exclusion_lower =
      get_exclusion(t->info()->mp_window, t->get_tile_row(), t->get_tile_col());
  partialcross_kern<true, true>(t->QT(), mpa.data(), mpia.data(), mpb.data(),
                                mpib.data(), t->dfa(), t->dga(), t->normsa(),
                                t->dfb(), t->dgb(), t->normsb(), width, height,
                                exclusion_lower, 0);
  combine_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()),
                  width);
  combine_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()),
                  height);
  return SCAMP_NO_ERROR;
}

SCAMPError_t cpu_kernel_self_join_lower(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);
  split_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()), width);
  split_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()), height);
  int exclusion_upper = get_exclusion(t->info()->mp_window, t->get_tile_col(),
                                      t->get_tile_row() + height);
  partialcross_kern<true, true>(t->QT(), mpb.data(), mpib.data(), mpa.data(),
                                mpia.data(), t->dfb(), t->dgb(), t->normsb(),
                                t->dfa(), t->dga(), t->normsa(), height, width,
                                0, exclusion_upper);
  combine_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()),
                  width);
  combine_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()),
                  height);
  return SCAMP_NO_ERROR;
}
SCAMPError_t cpu_kernel_ab_join_upper(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);
  split_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()), width);
  split_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()), height);
  std::pair<int, int> exclusion_pair(0, 0);
  if (t->info()->is_aligned) {
    int start_col = t->get_tile_col();
    int start_row = t->get_tile_row();
    if (t->info()->global_start_col_position >= 0 &&
        t->info()->global_start_row_position >= 0) {
      start_col += t->info()->global_start_col_position;
      start_row += t->info()->global_start_row_position;
    }
    exclusion_pair = get_exclusion_for_ab_join(t->info()->mp_window, start_row,
                                               start_col, true, width);
  }
  if (t->info()->computing_rows) {
    partialcross_kern<true, true>(
        t->QT(), mpa.data(), mpia.data(), mpb.data(), mpib.data(), t->dfa(),
        t->dga(), t->normsa(), t->dfb(), t->dgb(), t->normsb(), width, height,
        exclusion_pair.first, exclusion_pair.second);
  } else {
    partialcross_kern<false, true>(
        t->QT(), mpa.data(), mpia.data(), mpb.data(), mpib.data(), t->dfa(),
        t->dga(), t->normsa(), t->dfb(), t->dgb(), t->normsb(), width, height,
        exclusion_pair.first, exclusion_pair.second);
  }
  combine_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()),
                  width);
  combine_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()),
                  height);
  return SCAMP_NO_ERROR;
}
SCAMPError_t cpu_kernel_ab_join_lower(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);
  split_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()), width);
  split_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()), height);
  std::pair<int, int> exclusion_pair(0, 0);
  if (t->info()->is_aligned) {
    int start_col = t->get_tile_col();
    int start_row = t->get_tile_row();
    if (t->info()->global_start_col_position >= 0 &&
        t->info()->global_start_row_position >= 0) {
      start_col += t->info()->global_start_col_position;
      start_row += t->info()->global_start_row_position;
    }
    exclusion_pair = get_exclusion_for_ab_join(t->info()->mp_window, start_row,
                                               start_col, false, height);
  }

  if (t->info()->computing_rows) {
    partialcross_kern<true, true>(
        t->QT(), mpb.data(), mpib.data(), mpa.data(), mpia.data(), t->dfb(),
        t->dgb(), t->normsb(), t->dfa(), t->dga(), t->normsa(), height, width,
        exclusion_pair.first, exclusion_pair.second);
  } else {
    partialcross_kern<true, false>(
        t->QT(), mpb.data(), mpib.data(), mpa.data(), mpia.data(), t->dfb(),
        t->dgb(), t->normsb(), t->dfa(), t->dga(), t->normsa(), height, width,
        exclusion_pair.first, exclusion_pair.second);
  }
  combine_profile(mpa, mpia, reinterpret_cast<uint64_t*>(t->profile_a()),
                  width);
  combine_profile(mpb, mpib, reinterpret_cast<uint64_t*>(t->profile_b()),
                  height);
  return SCAMP_NO_ERROR;
}
};  // namespace SCAMP
