#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <thread>
#include "../src/SCAMP.h"
#include "../src/common.h"
#include "../src/scamp_utils.h"

namespace py = pybind11;

void SplitProfile1NNINDEX(const std::vector<uint64_t> profile,
                          std::vector<float>* NN, std::vector<int>* index,
                          bool output_pearson, int window) {
  NN->clear();
  index->clear();
  for (auto& elem : profile) {
    SCAMP::mp_entry e;
    e.ulong = elem;
    if (output_pearson) {
      NN->push_back(e.floats[0]);
    } else {
      NN->push_back(ConvertToEuclidean(e.floats[0], window));
    }
    index->push_back(e.ints[1]);
  }
}

std::vector<std::tuple<int64_t, int64_t, float>> SplitProfileKNN(
    std::vector<
        std::priority_queue<SCAMP::SCAMPmatch, std::vector<SCAMP::SCAMPmatch>,
                            SCAMP::compareMatch>>& matches,
    bool output_pearson, int window) {
  std::vector<std::tuple<int64_t, int64_t, float>> result;
  for (auto& pq : matches) {
    while (!pq.empty()) {
      float corr;
      if (output_pearson) {
        corr = pq.top().corr;
      } else {
        corr = ConvertToEuclidean(pq.top().corr, window);
      }
      result.emplace_back(pq.top().col, pq.top().row, corr);
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
  args.profile_a.default_thresh = 0;
  args.profile_b.default_thresh = 0;

  args.matrix_mode = false;
  return args;
}

void get_args_based_on_kwargs(SCAMP::SCAMPArgs* args, py::kwargs kwargs,
                              bool& pearson, std::vector<int>& gpus,
                              int& num_cpus) {
  for (auto item : kwargs) {
    std::string key = std::string(py::str(*item.first));
    if (key == "threshold") {
      args->distance_threshold = item.second.cast<double>();
      args->profile_a.default_thresh = args->distance_threshold;
      args->profile_b.default_thresh = args->distance_threshold;

      if (args->distance_threshold > 1 || args->distance_threshold < -1) {
        throw std::invalid_argument(
            "Invalid threshold specified: value must be between -1 and 1");
      }
    } else if (key == "verbose") {
      args->silent_mode = !item.second.cast<bool>();
    } else if (key == "mheight") {
      args->matrix_height = item.second.cast<int>();
      if (args->matrix_height <= 0) {
        throw std::invalid_argument(
            "Invalid matrix height specified: value must be greater than 0");
      }
    } else if (key == "mwidth") {
      args->matrix_width = item.second.cast<int>();
      if (args->matrix_width <= 0) {
        throw std::invalid_argument(
            "Invalid matrix width specified: value must be greater than 0");
      }
    } else if (key == "precision") {
      std::string ptype = item.second.cast<std::string>();
      if (ptype == "single") {
        args->precision_type = SCAMP::PRECISION_SINGLE;
      } else if (ptype == "mixed") {
        args->precision_type = SCAMP::PRECISION_MIXED;
      } else if (ptype == "double") {
        args->precision_type = SCAMP::PRECISION_DOUBLE;
      } else {
        throw std::invalid_argument(
            "Invalid precision type specified: valid options are single, "
            "mixed, double");
      }
    } else if (key == "pearson") {
      pearson = item.second.cast<bool>();
    } else if (key == "gpus") {
      gpus = item.second.cast<std::vector<int>>();
    } else if (key == "threads") {
      num_cpus = item.second.cast<int>();
      if (num_cpus < 0) {
        throw std::invalid_argument(
            "Invalid number of cpu worker threads specified, must be greater "
            "than or equal to 0.");
      }
    } else {
      throw std::invalid_argument(
          "Invalid keyword argument specified unknown argument: " + key);
    }
  }
  return;
}

bool setup_and_do_SCAMP(SCAMP::SCAMPArgs* args, py::kwargs kwargs) {
  std::vector<int> gpus;
  int num_cpus = 0;
  bool pearson = false;
  if (kwargs) {
    get_args_based_on_kwargs(args, kwargs, pearson, gpus, num_cpus);
  }
  InitProfileMemory(args);
  if (gpus.empty() && num_cpus == 0) {
    SCAMP::do_SCAMP(args);
  } else {
    SCAMP::do_SCAMP(args, gpus, num_cpus);
  }
  return pearson;
}

// 1NN_INDEX ab join
std::tuple<std::vector<float>, std::vector<int>> scamp(
    const std::vector<double>& a, const std::vector<double>& b, int m,
    const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  std::vector<float> NN;
  std::vector<int> index;
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, &NN, &index,
                       output_pearson, args.window);
  return std::make_tuple(NN, index);
}

// 1NN_INDEX self join
std::tuple<std::vector<float>, std::vector<int>> scamp(
    const std::vector<double>& a, int m, const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  std::vector<float> NN;
  std::vector<int> index;
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, &NN, &index,
                       output_pearson, args.window);
  return std::make_tuple(NN, index);
}

// KNN ab join
std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, const std::vector<double>& b, int m, int k,
    const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;
  args.max_matches_per_column = k;
  args.profile_type = SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  return SplitProfileKNN(args.profile_a.data[0].match_value, output_pearson,
                         args.window);
}

// KNN self join
std::vector<std::tuple<int64_t, int64_t, float>> scamp_knn(
    const std::vector<double>& a, int m, int k, const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;
  args.max_matches_per_column = k;
  args.profile_type = SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  return SplitProfileKNN(args.profile_a.data[0].match_value, output_pearson,
                         args.window);
}

// SUM self join
std::vector<double> scamp_sum(const std::vector<double>& a, int m,
                              const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;
  args.profile_type = SCAMP::PROFILE_TYPE_SUM_THRESH;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  return args.profile_a.data[0].double_value;
  // return SplitProfileKNN(args.profile_a.data[0].match_value, output_pearson,
  // args.window);
}

// SUM ab join
std::vector<double> scamp_sum(const std::vector<double>& a,
                              const std::vector<double>& b, int m,
                              const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;
  args.profile_type = SCAMP::PROFILE_TYPE_SUM_THRESH;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  return args.profile_a.data[0].double_value;
}

bool has_gpu_support() {
#ifdef _HAS_CUDA_
  return true;
#else
  return false;
#endif
}

bool (*GPU_supported)() = &has_gpu_support;
std::tuple<std::vector<float>, std::vector<int>> (*self_join_1NN_INDEX)(
    const std::vector<double>&, int, const py::kwargs&) = &scamp;
std::tuple<std::vector<float>, std::vector<int>> (*ab_join_1NN_INDEX)(
    const std::vector<double>&, const std::vector<double>&, int,
    const py::kwargs&) = &scamp;

std::vector<double> (*self_join_SUM_THRESH)(const std::vector<double>&, int,
                                            const py::kwargs&) = &scamp_sum;
std::vector<double> (*ab_join_SUM_THRESH)(const std::vector<double>&,
                                          const std::vector<double>&, int,
                                          const py::kwargs&) = &scamp_sum;

std::vector<std::tuple<int64_t, int64_t, float>> (*self_join_KNN)(
    const std::vector<double>&, int, int, const py::kwargs&) = &scamp_knn;
std::vector<std::tuple<int64_t, int64_t, float>> (*ab_join_KNN)(
    const std::vector<double>&, const std::vector<double>&, int, int,
    const py::kwargs&) = &scamp_knn;

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
        Returns the self-join matrix profile of a time series.
    )pbdoc");

  m.def("scamp", ab_join_1NN_INDEX, R"pbdoc(
        Returns the ab-join matrix profile of 2 time series.
    )pbdoc");

  m.def("scamp_sum", self_join_SUM_THRESH, R"pbdoc(
        Returns the sum of the correlations above specified threshold (default 0) for each subsequence in a time series.
    )pbdoc");

  m.def("scamp_sum", ab_join_SUM_THRESH, R"pbdoc(
        For each subsequence in time series a, returns the sum of the correlations to subsequences in time series b above specified threshold (default 0).
    )pbdoc");

  m.def("scamp_knn", self_join_KNN, R"pbdoc(
        Returns the k nearest neighbors for each subsequence in a time series)pbdoc");

  m.def("scamp_knn", ab_join_KNN, R"pbdoc(
        For each subsequence in time series A, returns its K nearest neighbors (approximate) in time series B)pbdoc");

  m.attr("__version__") = "dev";
}
