#include "cpu_stats.h"
#include <vector>
namespace SCAMP {

void compute_statistics_cpu(const google::protobuf::RepeatedField<double> &T,
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

  for (int i = 0; i < n; ++i) {
    means[i] = prefix_sum[i + m] - prefix_sum[i];
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
    norms[i] = 1.0 / sqrt(norms[i]);
  }

  for (int i = 0; i < n - 1; ++i) {
    df[i] = (T[i + m] - T[i]) / (double)2;
    dg[i] = (T[i + m] - means[i + 1]) + (T[i] - means[i]);
  }

  info->set(means, norms, df, dg);
}

}  // namespace SCAMP
