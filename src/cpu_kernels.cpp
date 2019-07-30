#include "cpu_kernels.h"
#include <vector>

namespace SCAMP {

// Kernel for computing matrix profiles on the CPU
// TODO(zpzim): This is unoptimized, we can get 3x+ additional throughput
// by performing optimizations
template <bool computing_rows, bool computing_cols>
static inline void partialcross_kern(
    double* __restrict cov, double* __restrict mpa, int* __restrict mpia,
    double* __restrict mpb, int* __restrict mpib, const double* __restrict dfa,
    const double* __restrict dga, const double* __restrict invna,
    const double* __restrict dfb, const double* __restrict dgb,
    const double* __restrict invnb, const int amx, const int bmx,
    const int amin, const int upper_excl) {
  for (int ia = amin; ia < amx - upper_excl + 1; ia++) {
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

void split_profile(std::vector<double>* mp, std::vector<int32_t>* mpi,
                   const uint64_t* profile, int len) {
  mp_entry e;
  for (int i = 0; i < len; ++i) {
    e.ulong = profile[i];
    mp->at(i) = static_cast<double>(e.floats[0]);
    mpi->at(i) = e.ints[1];
  }
}

void combine_profile(const std::vector<double>& mp,
                     const std::vector<int32_t>& mpi, uint64_t* profile,
                     int len) {
  mp_entry e;
  for (int i = 0; i < len; ++i) {
    e.floats[0] = static_cast<float>(mp[i]);
    e.ints[1] = mpi[i];
    profile[i] = e.ulong;
  }
}

// Self join on the upper triangular portion of the tile
SCAMPError_t cpu_kernel_self_join_upper(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  // TODO(zpzim): we should allocate these vectors in the tile constructor
  // We are taking some amount of perfomance penalty for allocating them here
  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);

  // TODO(zpzim): These splits should be done during the InitProfile method in
  // tile.cpp
  split_profile(&mpa, &mpia,
                reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                width);
  split_profile(&mpb, &mpib,
                reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                height);

  std::pair<int, int> exclusion = t->get_exclusion_for_self_join(true);
  partialcross_kern<true, true>(t->QT(), mpa.data(), mpia.data(), mpb.data(),
                                mpib.data(), t->dfa(), t->dga(), t->normsa(),
                                t->dfb(), t->dgb(), t->normsb(), width, height,
                                exclusion.first, exclusion.second);
  // TODO(zpzim): These combines should be done in the CopyProfileToHost method
  // in tile.cpp
  combine_profile(mpa, mpia,
                  reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                  width);
  combine_profile(mpb, mpib,
                  reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                  height);
  return SCAMP_NO_ERROR;
}

// Self join on the lower triangular portion of the tile
SCAMPError_t cpu_kernel_self_join_lower(Tile* t) {
  if (t->info()->profile_type != PROFILE_TYPE_1NN_INDEX) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }
  if (t->info()->fp_type != PRECISION_DOUBLE) {
    return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
  }

  size_t height = t->get_tile_height() - t->info()->mp_window + 1;
  size_t width = t->get_tile_width() - t->info()->mp_window + 1;

  // TODO(zpzim): we should allocate these vectors in the tile constructor
  // We are taking some amount of perfomance penalty for allocating them here
  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);

  // TODO(zpzim): These splits should be done during the InitProfile method in
  // tile.cpp
  split_profile(&mpa, &mpia,
                reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                width);
  split_profile(&mpb, &mpib,
                reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                height);

  std::pair<int, int> exclusion = t->get_exclusion_for_self_join(false);
  partialcross_kern<true, true>(t->QT(), mpb.data(), mpib.data(), mpa.data(),
                                mpia.data(), t->dfb(), t->dgb(), t->normsb(),
                                t->dfa(), t->dga(), t->normsa(), height, width,
                                exclusion.first, exclusion.second);
  // TODO(zpzim): These combines should be done in the CopyProfileToHost method
  // in tile.cpp
  combine_profile(mpa, mpia,
                  reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                  width);
  combine_profile(mpb, mpib,
                  reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
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

  // TODO(zpzim): we should allocate these vectors in the tile constructor
  // We are taking some amount of perfomance penalty for allocating them here
  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);

  // TODO(zpzim): These splits should be done during the InitProfile method in
  // tile.cpp
  split_profile(&mpa, &mpia,
                reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                width);
  split_profile(&mpb, &mpib,
                reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                height);

  std::pair<int, int> exclusion_pair = t->get_exclusion_for_ab_join(true);
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
  // TODO(zpzim): These combines should be done in the CopyProfileToHost method
  // in tile.cpp
  combine_profile(mpa, mpia,
                  reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                  width);
  combine_profile(mpb, mpib,
                  reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
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

  // TODO(zpzim): we should allocate these vectors in the tile constructor
  // We are taking some amount of perfomance penalty for allocating them here
  std::vector<double> mpa(width), mpb(height);
  std::vector<int32_t> mpia(width), mpib(height);

  // TODO(zpzim): These splits should be done during the InitProfile method in
  // tile.cpp
  split_profile(&mpa, &mpia,
                reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                width);
  split_profile(&mpb, &mpib,
                reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                height);
  std::pair<int, int> exclusion_pair = t->get_exclusion_for_ab_join(false);

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
  // TODO(zpzim): These combines should be done in the CopyProfileToHost method
  // in tile.cpp
  combine_profile(mpa, mpia,
                  reinterpret_cast<uint64_t*>(t->profile_a()),  // NOLINT
                  width);
  combine_profile(mpb, mpib,
                  reinterpret_cast<uint64_t*>(t->profile_b()),  // NOLINT
                  height);
  return SCAMP_NO_ERROR;
}
};  // namespace SCAMP
