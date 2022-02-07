#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <thread>
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_interface.h"
#include "common/scamp_utils.h"

namespace py = pybind11;

void SplitProfile1NNINDEX(const std::vector<uint64_t> profile,
                          py::array_t<float>& nn, py::array_t<int>& index,
                          bool output_pearson, int window) {
  auto nn_ptr = reinterpret_cast<float*>(nn.request().ptr);
  auto index_ptr = reinterpret_cast<int*>(index.request().ptr);
  int count = 0;
  for (auto& elem : profile) {
    SCAMP::mp_entry e;
    e.ulong = elem;
    if (output_pearson) {
      nn_ptr[count] = CleanupPearson(e.floats[0]);
    } else {
      nn_ptr[count] = ConvertToEuclidean(e.floats[0], window);
    }
    index_ptr[count] = e.ints[1];
    count++;
  }
}

std::vector<std::tuple<int64_t, int64_t, float>> SplitProfileKNN(
    std::vector<
        std::priority_queue<SCAMP::SCAMPmatch, std::vector<SCAMP::SCAMPmatch>,
                            SCAMP::compareMatch>>& matches,
    bool output_pearson, int window) {
  std::vector<std::tuple<int64_t, int64_t, float>> result;
  for (auto& pq : matches) {
    std::list<SCAMP::SCAMPmatch> elems;
    while (!pq.empty()) {
      elems.push_front(pq.top());
      pq.pop();
    }
    for (auto& elem : elems) {
      float corr;
      if (output_pearson) {
        corr = CleanupPearson(elem.corr);
      } else {
        corr = ConvertToEuclidean(elem.corr, window);
      }
      result.emplace_back(elem.col, elem.row, corr);
    }
  }
  return result;
}

template <typename T>
py::array_t<T> vec2pyarr(const std::vector<T>& arr, bool pearson = true,
                         int window = 0) {
  py::array_t<T> result(arr.size());
  auto ptr = reinterpret_cast<T*>(result.request().ptr);
  for (int i = 0; i < arr.size(); ++i) {
    if (pearson) {
      ptr[i] = CleanupPearson(arr[i]);
    } else {
      ptr[i] = ConvertToEuclidean(arr[i], window);
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

  return args;
}

bool KeyIsOkForProfileType(std::string key, SCAMP::SCAMPProfileType type) {
  static const std::set<std::string> nn_index = {"verbose", "precision",
                                                 "pearson", "gpus", "threads"};
  static const std::set<std::string> sum_thresh = {
      "verbose", "precision", "pearson", "gpus", "threads", "threshold"};
  static const std::set<std::string> knn = {
      "verbose", "precision", "pearson", "gpus", "threads", "threshold"};
  static const std::set<std::string> matrix = {
      "verbose", "precision", "pearson", "gpus",
      "threads", "threshold", "mheight", "mwidth"};

  switch (type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
      return nn_index.count(key) == 1;
    case SCAMP::PROFILE_TYPE_SUM_THRESH:
      return sum_thresh.count(key) == 1;
    case SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      return knn.count(key) == 1;
    case SCAMP::PROFILE_TYPE_MATRIX_SUMMARY:
      return matrix.count(key) == 1;
    default:
      return false;
  }
}

void get_args_based_on_kwargs(SCAMP::SCAMPArgs* args, py::kwargs kwargs,
                              bool& pearson, std::vector<int>& gpus,
                              int& num_cpus) {
  for (auto item : kwargs) {
    std::string key = std::string(py::str(*item.first));
    if (!KeyIsOkForProfileType(key, args->profile_type)) {
      throw std::invalid_argument(
          "Invalid keyword argument specified unknown argument: " + key);
    }
    if (key == "threshold") {
      args->distance_threshold = item.second.cast<double>();
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
      } else if (ptype == "ultra") {
        args->precision_type = SCAMP::PRECISION_ULTRA;
      } else {
        throw std::invalid_argument(
            "Invalid precision type specified: valid options are single, "
            "mixed, double, ultra");
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
  // If an empty list of GPUs was specified we should use CPU only.
  if (kwargs.contains("gpus") && gpus.empty()) {
    if (num_cpus <= 0) {
      num_cpus = std::thread::hardware_concurrency();
    }
    SCAMP::do_SCAMP(args, gpus, num_cpus);
    // If no threads/GPUs were specified, let SCAMP figure out what to do.
  } else if (gpus.empty() && num_cpus == 0) {
    SCAMP::do_SCAMP(args);
  } else {
    SCAMP::do_SCAMP(args, gpus, num_cpus);
  }
  return pearson;
}

// 1NN_INDEX ab join
std::tuple<py::array_t<float>, py::array_t<int>> scamp(
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

  py::array_t<float> result_nn(args.profile_a.data[0].uint64_value.size());
  py::array_t<int> result_index(args.profile_a.data[0].uint64_value.size());

  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, result_nn,
                       result_index, output_pearson, args.window);

  return std::make_tuple(result_nn, result_index);
}

// 1NN_INDEX self join
std::tuple<py::array_t<float>, py::array_t<int>> scamp(
    const std::vector<double>& a, int m, const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  py::array_t<float> result_nn(args.profile_a.data[0].uint64_value.size());
  py::array_t<int> result_index(args.profile_a.data[0].uint64_value.size());
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, result_nn,
                       result_index, output_pearson, args.window);
  return std::make_tuple(result_nn, result_index);
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
py::array_t<double> scamp_sum(const std::vector<double>& a, int m,
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

  return vec2pyarr<double>(args.profile_a.data[0].double_value);
}

// SUM ab join
py::array_t<double> scamp_sum(const std::vector<double>& a,
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

  return vec2pyarr<double>(args.profile_a.data[0].double_value);
}

py::array_t<float> scamp_matrix(const std::vector<double>& a, int m,
                                const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = a;
  args.window = m;
  args.has_b = false;
  args.computing_rows = true;
  args.computing_columns = true;
  args.profile_type = SCAMP::PROFILE_TYPE_MATRIX_SUMMARY;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  auto arr =
      vec2pyarr<float>(args.profile_a.data[0].float_value, output_pearson, m);
  arr.resize({args.matrix_height, args.matrix_width});
  return arr;
}

py::array_t<float> scamp_matrix(const std::vector<double>& a,
                                const std::vector<double>& b, int m,
                                const py::kwargs& kwargs) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.computing_rows = false;
  args.computing_columns = true;
  args.profile_type = SCAMP::PROFILE_TYPE_MATRIX_SUMMARY;
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;

  bool output_pearson = setup_and_do_SCAMP(&args, kwargs);

  auto arr =
      vec2pyarr<float>(args.profile_a.data[0].float_value, output_pearson, m);
  arr.resize({args.matrix_height, args.matrix_width});
  return arr;
}

bool has_gpu_support() { return SCAMP::num_available_gpus() > 0; }

bool (*GPU_supported)() = &has_gpu_support;
std::tuple<py::array_t<float>, py::array_t<int>> (*self_join_1NN_INDEX)(
    const std::vector<double>&, int, const py::kwargs&) = &scamp;
std::tuple<py::array_t<float>, py::array_t<int>> (*ab_join_1NN_INDEX)(
    const std::vector<double>&, const std::vector<double>&, int,
    const py::kwargs&) = &scamp;

py::array_t<double> (*self_join_SUM_THRESH)(const std::vector<double>&, int,
                                            const py::kwargs&) = &scamp_sum;
py::array_t<double> (*ab_join_SUM_THRESH)(const std::vector<double>&,
                                          const std::vector<double>&, int,
                                          const py::kwargs&) = &scamp_sum;

py::array_t<float> (*self_join_MATRIX)(const std::vector<double>&, int,
                                       const py::kwargs&) = &scamp_matrix;
py::array_t<float> (*ab_join_MATRIX)(const std::vector<double>&,
                                     const std::vector<double>&, int,
                                     const py::kwargs&) = &scamp_matrix;

std::vector<std::tuple<int64_t, int64_t, float>> (*self_join_KNN)(
    const std::vector<double>&, int, int, const py::kwargs&) = &scamp_knn;
std::vector<std::tuple<int64_t, int64_t, float>> (*ab_join_KNN)(
    const std::vector<double>&, const std::vector<double>&, int, int,
    const py::kwargs&) = &scamp_knn;

PYBIND11_MODULE(pyscamp, m) {
  m.doc() = R"pbdoc(
        pyscamp: Python bindings for SCAMP
        ----------------------------------

        .. currentmodule:: pyscamp

        .. autosummary::
           :toctree: _generate

           selfjoin
           abjoin
           selfjoin_sum
           abjoin_sum
           selfjoin_knn
           abjoin_knn
           selfjoin_matrix
           abjoin_matrix 
    )pbdoc";

  m.def("gpu_supported", GPU_supported, R"pbdoc(
        Returns true if both 1) The module was compiled with GPU support and 2) GPUs are available.
        )pbdoc");

  m.def("selfjoin", self_join_1NN_INDEX, py::arg("a"), py::arg("m"), R"pbdoc(
    Computes the matrix profile for time series A.
  
    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :return: A tuple containing the matrix profile as the first element and the indices as a the second element.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    )pbdoc");

  m.def("abjoin", ab_join_1NN_INDEX, py::arg("a"), py::arg("b"), py::arg("m"),
        R"pbdoc(
    For each subsequence in time series A, finds the nearest neighbor in time series B.

    :param a: Time series, b will be queried for subsequences in a.
    :type a: 1D array 
    :param b: Time series in which to search for matches for subsequences in a.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :return: A tuple. First element: The nearest neighbor distance of subsequences in a to time series b. Second element: The index (in b) of each nearest neighbor.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    )pbdoc");

  m.def("selfjoin_sum", self_join_SUM_THRESH, py::arg("a"), py::arg("m"),
        R"pbdoc(
    Returns the sum of the correlations above specified threshold (default 0) for each subsequence in a time series.

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: For each subsequence in A, returns the sum of correlations above the the specified threshold to other subesequences in A.
    :rtype: np.ndarray[float64]
    )pbdoc");

  m.def("abjoin_sum", ab_join_SUM_THRESH, py::arg("a"), py::arg("b"),
        py::arg("m"), R"pbdoc(
    For each subsequence in time series a, returns the sum of the correlations to subsequences in time series b above specified threshold (default 0).

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param b: Time series to search for matches.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: For each subsequence in A, returns the sum of correlations above the the specified threshold in B.
    :rtype: np.ndarray[float64]
    )pbdoc");

  m.def("selfjoin_knn", self_join_KNN, py::arg("a"), py::arg("m"), py::arg("k"),
        R"pbdoc(
    [GPU ONLY, EXPERIMENTAL] Returns the approximate k nearest neighbors for each subsequence in a time series

    :param a: Time series to compute the KNN matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param k: Number of neighbors to return for each subsequence
    :type k: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: List of tuples (col, row, distance) containing the matches (up to K) for each column of the distance matrix, row is the index of the match, and d is the distance between the two subsequences
    :rtype: List of tuple[int, int, float]
    )pbdoc");

  m.def("abjoin_knn", ab_join_KNN, py::arg("a"), py::arg("b"), py::arg("m"),
        py::arg("k"), R"pbdoc(
    [GPU ONLY, EXPERIMENTAL] For each subsequence in time series A, returns its Approximate K nearest neighbors in time series B

    :param a: Time series to compute the KNN matrix profile for.`
    :type a: 1D array
    :param b: Time series in which to search for matches.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param k: Number of neighbors to return for each subsequence
    :type k: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: List of tuples (col, row, distance) containing the matches (up to K) for each column of the distance matrix, col is the index in A, row is the index in B of the match, and d is the distance between the two subsequences
    :rtype: List of tuple[int, int, float]
    )pbdoc");

  m.def("selfjoin_matrix", self_join_MATRIX, py::arg("a"), py::arg("m"),
        R"pbdoc(
    [EXPERIMENTAL] Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidian Distance

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param mheight: Height of the pooled distance matrix to output. Default 50
    :type mheight: int, optional
    :param mwidth: Width of the pooled distance matrix to output. Default 50
    :type mwidth: int, optional
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: A 2D array of height of mheight and width of mwidth. This is a pooled version of the full distance matrix.
    :rtype: 2D array
    )pbdoc");

  m.def("abjoin_matrix", ab_join_MATRIX, py::arg("a"), py::arg("b"),
        py::arg("m"), R"pbdoc(
    [EXPERIMENTAL] Returns a pooled version of the distance matrix with HxW of [mheight x mwidth], pooling operation is max() for Pearson Correlation and min() for Euclidian Distance

    :param a: Time series corresponding to the columns of the distance matrix.
    :type a: 1D array
    :param b: Time series corresponding to the rows of the distance matrix.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param mheight: Height of the pooled distance matrix to output. Default 50
    :type mheight: int, optional
    :param mwidth: Width of the pooled distance matrix to output. Default 50
    :type mwidth: int, optional
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :return: A 2D array of height of mheight and width of mwidth. This is a pooled version of the full distance matrix.
    :rtype: 2D array
)pbdoc");

  m.attr("__version__") = "dev";
}
