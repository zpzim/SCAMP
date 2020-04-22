#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <thread>
#include "../src/SCAMP.h"
#include "../src/common.h"
#include "../src/scamp_utils.h"

void SplitProfile1NNINDEX(const std::vector<uint64_t> profile,
                          std::vector<float>* NN, std::vector<int>* index) {
  NN->clear();
  index->clear();
  for (auto& elem : profile) {
    SCAMP::mp_entry e;
    e.ulong = elem;
    NN->push_back(e.floats[0]);
    index->push_back(e.ints[1]);
  }
}

std::vector<std::tuple<int64_t, int64_t, float>> SplitProfileKNN(
    std::vector<
        std::priority_queue<SCAMP::SCAMPmatch, std::vector<SCAMP::SCAMPmatch>,
                            SCAMP::compareMatch>>& matches) {
  std::vector<std::tuple<int64_t, int64_t, float>> result;
  for (auto& pq : matches) {
    while (!pq.empty()) {
      result.emplace_back(pq.top().col, pq.top().row, pq.top().corr);
      pq.pop();
    }
  }
  return result;
}

SCAMP::SCAMPArgs GetDefaultSCAMPArgs() {
  auto profile_type = SCAMP::PROFILE_TYPE_1NN_INDEX;
  SCAMP::SCAMPArgs args;
  args.has_b = false;
  args.max_tile_size = 128000;
  args.distributed_start_row = -1;
  args.distributed_start_col = -1;
  args.distance_threshold = 0;
  args.precision_type = SCAMP::PRECISION_DOUBLE;
  args.profile_type = profile_type;
  args.computing_rows = true;
  args.computing_columns = true;
  args.keep_rows_separate = false;
  args.is_aligned = false;
  args.silent_mode = true;
  args.max_matches_per_column = 5;
  args.matrix_height = 50;
  args.matrix_width = 50;
  args.profile_a.type = profile_type;
  args.profile_b.type = profile_type;
  args.profile_a.matrix_height = -1;
  args.profile_b.matrix_height = -1;
  args.profile_a.matrix_width = -1;
  args.profile_b.matrix_width = -1;
  args.profile_a.output_matrix = false;
  args.profile_b.output_matrix = false;
  args.matrix_mode = false;
  return args;
}

std::tuple<std::vector<float>, std::vector<int>> scamp(
    const std::vector<double>& a, const std::vector<double>& b, int m) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;

  InitProfileMemory(&args);
  SCAMP::do_SCAMP(&args);
  std::vector<float> NN;
  std::vector<int> index;
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, &NN, &index);
  return std::make_tuple(NN, index);
}

std::tuple<std::vector<float>, std::vector<int>> scamp(
    const std::vector<double>& a, int m) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;

  InitProfileMemory(&args);
  SCAMP::do_SCAMP(&args);
  std::vector<float> NN;
  std::vector<int> index;
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, &NN, &index);
  return std::make_tuple(NN, index);
}

std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, const std::vector<double>& b, int m, int k,
    double threshold) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;
  args.max_matches_per_column = k;
  args.distance_threshold = threshold;
  args.profile_type = SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  InitProfileMemory(&args);
  SCAMP::do_SCAMP(&args);
  return SplitProfileKNN(args.profile_a.data[0].match_value);
}

std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, const std::vector<double>& b, int m, int k) {
  return scamp_knn(a, b, m, k, 0);
}

std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, int m, int k, double threshold) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;
  args.max_matches_per_column = k;
  args.distance_threshold = threshold;
  args.profile_type = SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  InitProfileMemory(&args);
  SCAMP::do_SCAMP(&args);
  return SplitProfileKNN(args.profile_a.data[0].match_value);
}

std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, int m, int k) {
  return scamp_knn(a, m, k, 0);
}

bool has_gpu_support() {
#ifdef _HAS_CUDA_
  return true;
#else
  return false;
#endif
}

namespace py = pybind11;

bool (*GPU_supported)() = &has_gpu_support;
std::tuple<std::vector<float>, std::vector<int>> (*self_join_1NN_INDEX)(
    const std::vector<double>&, int) = &scamp;
std::tuple<std::vector<float>, std::vector<int>> (*ab_join_1NN_INDEX)(
    const std::vector<double>&, const std::vector<double>&, int) = &scamp;
std::vector<std::tuple<int64_t, int64_t, float>> (*self_join_KNN_thresh)(
    const std::vector<double>&, int, int, double) = &scamp_knn;
std::vector<std::tuple<int64_t, int64_t, float>> (*self_join_KNN)(
    const std::vector<double>&, int, int) = &scamp_knn;
std::vector<std::tuple<int64_t, int64_t, float>> (*ab_join_KNN_thresh)(
    const std::vector<double>&, const std::vector<double>&, int, int,
    double) = &scamp_knn;
std::vector<std::tuple<int64_t, int64_t, float>> (*ab_join_KNN)(
    const std::vector<double>&, const std::vector<double>&, int,
    int) = &scamp_knn;

PYBIND11_MODULE(pyscamp, m) {
  m.doc() = R"pbdoc(
        SCAMP: SCAlable Matrix Profile
        -------------------------------
        .. currentmodule:: scamp
        .. autosummary::
           :toctree: _generate
           SCAMP_AB
           SCAMP_SELF
    )pbdoc";

  m.def("gpu_supported", GPU_supported, R"pbdoc(
        Returns whether or not the module was compiled with GPU support)pbdoc");

  m.def("scamp", self_join_1NN_INDEX, R"pbdoc(
        Returns the self-join matrix profile of a time series (in Pearson Correlation)
    )pbdoc");

  m.def("scamp", ab_join_1NN_INDEX, R"pbdoc(
        Returns the ab-join matrix profile of 2 time series (in Pearson Correlation)
    )pbdoc");

  m.def("scamp_knn", self_join_KNN, R"pbdoc(
        Returns the k nearest neighbors for each subsequence in a time series)pbdoc");

  m.def("scamp_knn", self_join_KNN_thresh, R"pbdoc(
        Returns the k nearest neighbors for each subsequence in a time series, ignoring matches below 'threshold' correlation)pbdoc");

  m.def("scamp_knn", ab_join_KNN, R"pbdoc(
        For each subsequence in time series A, returns its K nearest neighbors in time series B)pbdoc");

  m.def("scamp_knn", ab_join_KNN_thresh, R"pbdoc(
        For each subsequence in time series A, returns its K nearest neighbors in time series B, ignoring matches below 'threshold' correlation)pbdoc");

  m.attr("__version__") = "dev";
}
