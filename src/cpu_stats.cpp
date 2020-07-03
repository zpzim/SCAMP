#include "cpu_stats.h"
#include <cmath>
#include <vector>
namespace SCAMP {

constexpr const double FLATNESS_EPSILON = 1e-7;

// Converts any NaN or inf values in the input to 0, returns the cleaned
// timeseries in timeseries_clean, and returns the subsequences which contained
// NaN in nanvals
void convert_non_finite_to_zero(const std::vector<double> &T, const int m,
                                std::vector<double> *timeseries_clean,
                                std::vector<bool> *nanvals) {
  timeseries_clean->resize(T.size());
  nanvals->resize(T.size() - m + 1);
  size_t steps_since_last_nan = m;
  for (int i = 0; i < T.size(); ++i) {
    if (std::isfinite(T[i])) {
      timeseries_clean->at(i) = T[i];
    } else {
      steps_since_last_nan = 0;
      timeseries_clean->at(i) = 0;
    }
    if (i >= m - 1) {
      nanvals->at(i - m + 1) = steps_since_last_nan < m;
    }
    steps_since_last_nan++;
  }
}

// moving mean over sequence a with window length w
// based on Ogita et. al, Accurate Sum and Dot Product

// A major source of rounding error is accumulated error in the mean values, so
// we use this to compensate. While the error bound is still a function of the
// conditioning of a very long dot product, we have observed a reduction of 3 -
// 4 digits lost to numerical roundoff when compared to older solutions.
std::vector<double> moving_mean(std::vector<double> a, int w) {
  std::vector<double> result(a.size() - w + 1, 0);
  double p = a[0];
  double s = 0;
  for (int i = 1; i < w; ++i) {
    double x = p + a[i];
    double z = x - p;
    s += (p - (x - z)) + (a[i] - z);
    p = x;
  }
  result[0] = (p + s) / w;

  for (int i = w; i < a.size(); ++i) {
    double x = p - a[i - w];
    double z = x - p;
    s += (p - (x - z)) - (a[i - w] + z);
    p = x;

    x = p + a[i];
    z = x - p;
    s += (p - (x - z)) + (a[i] - z);
    p = x;
    result[i - w + 1] = (p + s) / w;
  }
  return result;
}

// Computes the subsequence norms for a time series, subsequence length w

// This is an accurate, but slower O(len(a) * w) brute force method. (the
// default) There is room for optimization here either through unrolling and
// using SIMD operations, via multithreading, or using the GPU (if available)
std::vector<double> brute_force_sum_of_squared_difference(
    const std::vector<double> &a, const std::vector<double> &means, int w) {
  std::vector<double> result(a.size() - w + 1);
  for (int i = 0; i < a.size() - w + 1; ++i) {
    double sum = 0;
    for (int j = 0; j < w; ++j) {
      double val = a[i + j] - means[i];
      sum += val * val;
    }
    result[i] = sum;
  }
  return result;
}

// Computes the subesequence norms for a time series, subsequence length w

// This is a faster, less accurate method that should only be used if the
// slower method becomes a botteneck.

// This method can be enabled by using the high_precision_precompute flag
// in SCAMPArgs.
std::vector<double> fast_sum_of_squared_difference(
    const std::vector<double> &a, const std::vector<double> &means, int w) {
  std::vector<double> result(a.size() - w + 1);
  double sum = 0;
  for (int i = 0; i < w; ++i) {
    double val = a[i] - means[0];
    sum += val * val;
  }
  result[0] = sum;

  for (int i = 1; i < a.size() - w + 1; ++i) {
    result[i] = result[i - 1] +
                ((a[i - 1] - means[i - 1]) + (a[i + w - 1] - means[i])) *
                    (a[i + w - 1] - a[i - 1]);
  }

  return result;
}

// Computes all required statistics for SCAMP, populating info with these values
void compute_statistics_cpu(const std::vector<double> &T,
                            const std::vector<bool> &nanvalues,
                            PrecomputedInfo *info, size_t m,
                            bool high_precision_norms) {
  std::vector<double> prefix_sum(T.size());
  int n = T.size() - m + 1;
  std::vector<double> norms, df(n), dg(n);

  std::vector<double> means = moving_mean(T, m);

  if (high_precision_norms) {
    norms = brute_force_sum_of_squared_difference(T, means, m);
  } else {
    norms = fast_sum_of_squared_difference(T, means, m);
  }

  for (int i = 0; i < n; ++i) {
    // If the subsequence includes a NaN, we define the norm as NaN
    if (nanvalues[i]) {
      norms[i] = std::nan("NaN");
      // Check if the average distance from the mean is too small and this
      // subsequence should be considered FLAT
    } else if (norms[i] / m <= FLATNESS_EPSILON) {
      norms[i] = std::nan("NaN");
    } else {
      // Compute the inverse norm from the sum of squared differences
      norms[i] = static_cast<double>(1.0) / std::sqrt(norms[i]);
    }
  }

  for (int i = 0; i < n - 1; ++i) {
    df[i] = (T[i + m] - T[i]) / static_cast<double>(2);
    dg[i] = (T[i + m] - means[i + 1]) + (T[i] - means[i]);
  }

  info->set(means, norms, df, dg);
}

}  // namespace SCAMP
