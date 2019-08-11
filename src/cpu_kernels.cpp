#include "cpu_kernels.h"
#include <array>
#include <vector>
namespace SCAMP {

// this is hard coded for now. It needs to be a power of 2
// It may be desirable to make this larger, since the compiler is generally
// unable to fold everything into register names with either int or long long
// based indexing.
constexpr int unrollWid{32};

// set here for now. Could be set by cmake depending on what is supported. AVX
// and AVX2 benefit from at least 32 particularly if masked movement
// instructions are generated for the reduction steps.
constexpr int simdByteLen{32};

template <bool computing_rows, bool computing_cols>
void partialcross_kern(double* __restrict cov, double* __restrict mpa,
                       int* __restrict mpia, double* __restrict mpb,
                       int* __restrict mpib, const double* __restrict dfa,
                       const double* __restrict dga,
                       const double* __restrict invna,
                       const double* __restrict dfb,
                       const double* __restrict dgb,
                       const double* __restrict invnb, const int amx,
                       const int bmx, const int amin, const int upper_excl) {
  for (int ia = amin; ia < amx - upper_excl; ia += unrollWid) {
    int rowIters = std::min(amx - ia, bmx);
    int fullRowIters = std::max(0, std::min(amx - ia - unrollWid + 1, bmx));
    for (int ib = 0; ib < fullRowIters; ib++) {
      alignas(simdByteLen) std::array<double, unrollWid> corr;
      for (int diag = 0; diag < unrollWid; diag++) {
        double correlation = cov[ia + diag] * invna[ia + diag + ib] * invnb[ib];
        corr[diag] = std::isfinite(correlation) ? correlation : -2.0;
      }
      if (computing_cols) {
        for (int diag = 0; diag < unrollWid; diag++) {
          mpia[ia + diag + ib] =
              mpa[ia + diag + ib] >= corr[diag] ? mpia[ia + diag + ib] : ib;
          mpa[ia + diag + ib] = mpa[ia + diag + ib] >= corr[diag]
                                    ? mpa[ia + diag + ib]
                                    : corr[diag];
        }
      }
      if (computing_rows) {
        std::array<int, unrollWid / 2> corrIdx;
        for (int i = 0; i < unrollWid / 2; i++) {
          corrIdx[i] =
              corr[i] >= corr[i + unrollWid / 2] ? i : i + unrollWid / 2;
          corr[i] = corr[i] >= corr[i + unrollWid / 2]
                        ? corr[i]
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
        mpib[ib] = mpb[ib] >= corr[0] ? mpib[ib] : corrIdx[0] + ia + ib;
        mpb[ib] = mpb[ib] >= corr[0] ? mpb[ib] : corr[0];
      }
      for (int diag = 0; diag < unrollWid; diag++) {
        cov[ia + diag] += dfa[ia + diag + ib] * dgb[ib];
        cov[ia + diag] += dfb[ib] * dga[ia + diag + ib];
      }
    }
    for (int ib = fullRowIters; ib < rowIters; ib++) {
      int diagmax =
          std::min(std::min(amx - ia - upper_excl, amx - ia - ib), unrollWid);
      for (int diag = 0; diag < diagmax; diag++) {
        double corr = cov[ia + diag] * invna[ia + diag + ib] * invnb[ib];
        corr = std::isfinite(corr) ? corr : -2.0;
        if (computing_cols) {
          if (mpa[ia + diag + ib] < corr) {
            mpa[ia + diag + ib] = corr;
            mpia[ia + diag + ib] = ib;
          }
        }
        if (computing_rows) {
          if (mpb[ib] < corr) {
            mpb[ib] = corr;
            mpib[ib] = ia + diag + ib;
          }
        }
      }
      for (int diag = 0; diag < diagmax; diag++) {
        cov[ia + diag] += dfa[ia + diag + ib] * dgb[ib];
        cov[ia + diag] += dga[ia + diag + ib] * dfb[ib];
      }
    }
  }
}
/*
// Kernel for computing matrix profiles on the CPU
// Reference Implementation
template <bool computing_rows, bool computing_cols>
void partialcross_kern(
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
*/
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
