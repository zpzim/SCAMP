#include "common.h"
#include "tile.h"

namespace SCAMP {

static constexpr int NUM_EXTRA_OPERANDS = 3;

template <typename T>
struct SCAMPKernelInputArgs {
  SCAMPKernelInputArgs(Tile *t, bool transpose, bool ab_join);
  T *__restrict__ cov;
  const T *__restrict__ dfa;
  const T *__restrict__ dfb;
  const T *__restrict__ dga;
  const T *__restrict__ dgb;
  const T *__restrict__ normsa;
  const T *__restrict__ normsb;
  const T *__restrict__ extras[NUM_EXTRA_OPERANDS];
  unsigned long long int *profile_a_length;
  unsigned long long int *profile_b_length;
  int64_t max_matches_per_tile;
  int32_t n_x;
  int32_t n_y;
  int32_t exclusion_lower;
  int32_t exclusion_upper;
  OptionalArgs opt;
  void Print();
};

// Outputs an 'initial' distance value based on the type of profile being
// computed
template <typename DISTANCE_TYPE, SCAMPProfileType type>
__device__ inline DISTANCE_TYPE init_dist() {
  switch (type) {
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_1NN:
      // Smallest value possible is -1 so set to -2
      return static_cast<DISTANCE_TYPE>(-2);
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
    default:
      // We must set to 0 so we get an accurate sum
      return static_cast<DISTANCE_TYPE>(0);
  }
}

}  // namespace SCAMP
