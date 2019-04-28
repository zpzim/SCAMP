#include "cpu_stats.h"
#include <cmath>
#include <vector>
namespace SCAMP {

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

// Computes all required statistics for SCAMP, populating info with these values
void compute_statistics_cpu(const std::vector<double> &T,
                            const std::vector<bool> &nanvalues,
                            PrecomputedInfo *info, size_t m) {
  std::vector<double> prefix_sum(T.size());
  std::vector<double> prefix_sum_sq(T.size());
  int n = T.size() - m + 1;
  std::vector<double> norms(n), means(n), df(n), dg(n);

  prefix_sum[0] = T[0];
  prefix_sum_sq[0] = T[0] * T[0];
  for (int i = 1; i < T.size(); ++i) {
    prefix_sum[i] = T[i] + prefix_sum[i - 1];
    prefix_sum_sq[i] = T[i] * T[i] + prefix_sum_sq[i - 1];
  }

  means[0] = prefix_sum[m - 1] / static_cast<double>(m);
  for (int i = 1; i < n; ++i) {
    means[i] =
        (prefix_sum[i + m - 1] - prefix_sum[i - 1]) / static_cast<double>(m);
  }

  double sum = 0;
  for (int i = 0; i < m; ++i) {
    double val = T[i] - means[0];
    sum += val * val;
  }
  norms[0] = sum;

  for (int i = 1; i < n; ++i) {
    norms[i] =
        norms[i - 1] + ((T[i - 1] - means[i - 1]) + (T[i + m - 1] - means[i])) *
                           (T[i + m - 1] - T[i - 1]);
  }
  for (int i = 0; i < n; ++i) {
    if (nanvalues[i]) {
      norms[i] = std::nan("NaN");
    } else {
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
