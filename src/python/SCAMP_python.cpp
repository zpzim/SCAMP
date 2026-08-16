#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <thread>
#include "common/common.h"
#include "common/scamp_args.h"
#include "common/scamp_interface.h"
#include "common/scamp_utils.h"
#ifdef _HAS_CUDA_
#include <cuda_runtime.h>
#include "core/gpu_kernel/autotune.h"
#include "core/gpu_kernel/autotune_bench.h"
#endif

namespace py = pybind11;

// Convert a numpy array to a std::vector<double>, going through the buffer
// protocol with an explicit double dtype request and forcecast.  We avoid
// pybind11's automatic numpy ndarray -> std::vector<double> conversion (via
// pybind11/stl.h's list_caster) because it iterates the array as a Python
// sequence and round-trips every element through Python objects.  That path
// has been observed to silently produce zeroed inputs under NumPy >= 2 with
// older pybind11 releases, which in turn makes SCAMP compute on a degenerate
// (zero-variance) input and return a constant matrix profile (#129).
//
// Going through the buffer protocol with `forcecast` is dtype-safe regardless
// of the user's NumPy version: any 1D numeric array (float32, int64, etc.)
// is silently up-cast to float64 with a contiguous copy if needed, then
// memcpy'd into the std::vector.  Behaviour is identical on NumPy 1.x and 2.x.
static std::vector<double> ArrayToDoubleVector(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    const char* arg_name) {
  auto buf = arr.request();
  if (buf.ndim != 1) {
    throw std::invalid_argument(
        std::string("Argument '") + arg_name +
        "' must be a 1D array (got ndim=" + std::to_string(buf.ndim) + ").");
  }
  const double* data = static_cast<const double*>(buf.ptr);
  return std::vector<double>(data, data + buf.size);
}

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
  static const std::set<std::string> nn_index = {
      "verbose", "precision",           "pearson",      "gpus",
      "threads", "allow_trivial_match", "max_tile_size"};
  static const std::set<std::string> sum_thresh = {
      "verbose",   "precision",           "pearson",      "gpus", "threads",
      "threshold", "allow_trivial_match", "max_tile_size"};
  static const std::set<std::string> knn = {
      "verbose",   "precision",           "pearson",      "gpus", "threads",
      "threshold", "allow_trivial_match", "max_tile_size"};
  static const std::set<std::string> matrix = {
      "verbose",      "precision", "pearson",
      "gpus",         "threads",   "threshold",
      "mheight",      "mwidth",    "allow_trivial_match",
      "max_tile_size"};

  switch (type) {
    case SCAMP::PROFILE_TYPE_1NN_INDEX:
    case SCAMP::PROFILE_TYPE_1NN:
      // 1NN (no index) accepts the same options as 1NN_INDEX.
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
    } else if (key == "max_tile_size") {
      args->max_tile_size = item.second.cast<int>();
      if (args->max_tile_size <= 0) {
        throw std::invalid_argument(
            "Invalid max_tile_size specified: value must be greater than 0");
      }
    } else if (key == "precision") {
      std::string ptype = item.second.cast<std::string>();
      if (ptype == "single") {
        args->precision_type = SCAMP::PRECISION_SINGLE;
      } else if (ptype == "double") {
        args->precision_type = SCAMP::PRECISION_DOUBLE;
      } else if (ptype == "ultra") {
        args->precision_type = SCAMP::PRECISION_ULTRA;
      } else {
        throw std::invalid_argument(
            "Invalid precision type specified: valid options are single, "
            "double, ultra");
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
    } else if (key == "allow_trivial_match") {
      if (!args->has_b) {
        throw std::invalid_argument(
            "allow_trivial_match is only valid for ab-joins; self-joins always "
            "exclude trivial matches.");
      }
      args->is_aligned = !item.second.cast<bool>();
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
  // Determine if a GPU is used for execution.
  bool gpu_used = false;
  if (kwargs && kwargs.contains("gpus")) {
    gpu_used = !gpus.empty();
  } else if (kwargs && kwargs.contains("threads") && num_cpus > 0) {
    gpu_used = false;
  } else {
    gpu_used = (SCAMP::num_available_gpus() > 0);
  }

  // If a GPU is used and max_tile_size was not explicitly specified,
  // set it to 512k (512000) by default.
  if (gpu_used && (!kwargs || !kwargs.contains("max_tile_size"))) {
    args->max_tile_size = 512000;
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

// ===========================================================================
// Unified join entry point backing pyscamp.join().
//
// do_join() sets up a single SCAMPArgs from explicit structural arguments
// (method, want_index, left_right) rather than one hardcoded profile type per
// function, runs it, and returns a py::dict whose keys map onto JoinResult
// fields in the Python layer. This is what lets a single Python join() cover
// self/ab x {1nn, sum, knn, matrix} x index on/off x left/right in one place,
// including the two capabilities the per-function bindings never exposed:
// PROFILE_TYPE_1NN (index=False) and keep_rows_separate (left/right in one
// pass). The legacy selfjoin/abjoin/*_sum/*_knn/*_matrix functions are left
// untouched for backward compatibility.
// ===========================================================================

// --- per-direction extractors: one SCAMP::Profile -> numpy object(s) ---

py::array_t<float> extract_1nn(const SCAMP::Profile& p, bool pearson,
                               int window) {
  return vec2pyarr<float>(p.data[0].float_value, pearson, window);
}

std::pair<py::array_t<float>, py::array_t<int>> extract_1nn_index(
    const SCAMP::Profile& p, bool pearson, int window) {
  size_t n = p.data[0].uint64_value.size();
  py::array_t<float> nn(n);
  py::array_t<int> idx(n);
  SplitProfile1NNINDEX(p.data[0].uint64_value, nn, idx, pearson, window);
  return {std::move(nn), std::move(idx)};
}

py::array_t<double> extract_sum(const SCAMP::Profile& p) {
  return vec2pyarr<double>(p.data[0].double_value);
}

py::array_t<float> extract_matrix(const SCAMP::Profile& p, bool pearson,
                                  int window, int height, int width) {
  auto arr = vec2pyarr<float>(p.data[0].float_value, pearson, window);
  arr.resize({height, width});
  return arr;
}

// KNN as three parallel numpy arrays (col, row, distance), grouped by column
// and ordered like SplitProfileKNN (drains the per-column priority queues).
// Mutates p (empties the queues); do_join owns a local Profile so this is safe.
std::tuple<py::array_t<int64_t>, py::array_t<int64_t>, py::array_t<float>>
extract_knn(SCAMP::Profile& p, bool pearson, int window) {
  std::vector<SCAMP::SCAMPmatch> flat;
  for (auto& pq : p.data[0].match_value) {
    std::list<SCAMP::SCAMPmatch> elems;
    while (!pq.empty()) {
      elems.push_front(pq.top());
      pq.pop();
    }
    for (auto& e : elems) flat.push_back(e);
  }
  size_t n = flat.size();
  py::array_t<int64_t> cols(n), rows(n);
  py::array_t<float> dist(n);
  auto* c = cols.mutable_data();
  auto* r = rows.mutable_data();
  auto* d = dist.mutable_data();
  for (size_t i = 0; i < n; ++i) {
    c[i] = flat[i].col;
    r[i] = flat[i].row;
    d[i] = pearson ? CleanupPearson(flat[i].corr)
                   : ConvertToEuclidean(flat[i].corr, window);
  }
  return {std::move(cols), std::move(rows), std::move(dist)};
}

SCAMP::SCAMPProfileType ProfileTypeForMethod(const std::string& method,
                                             bool want_index) {
  if (method == "1nn")
    return want_index ? SCAMP::PROFILE_TYPE_1NN_INDEX : SCAMP::PROFILE_TYPE_1NN;
  if (method == "sum") return SCAMP::PROFILE_TYPE_SUM_THRESH;
  if (method == "knn") return SCAMP::PROFILE_TYPE_APPROX_ALL_NEIGHBORS;
  if (method == "matrix") return SCAMP::PROFILE_TYPE_MATRIX_SUMMARY;
  throw std::invalid_argument("do_join: unknown method '" + method + "'");
}

// Write one direction's outputs into `out` under the given key prefix
// ("" for the single-direction case, "left_"/"right_" for keep_rows).
void PutDirection(py::dict& out, const std::string& prefix,
                  const std::string& method, bool want_index, bool pearson,
                  int window, int height, int width, SCAMP::Profile& profile) {
  if (method == "1nn") {
    if (want_index) {
      auto pr = extract_1nn_index(profile, pearson, window);
      out[py::str(prefix + "profile")] = std::move(pr.first);
      out[py::str(prefix + "index")] = std::move(pr.second);
    } else {
      out[py::str(prefix + "profile")] = extract_1nn(profile, pearson, window);
    }
  } else if (method == "sum") {
    out[py::str(prefix + "profile")] = extract_sum(profile);
  } else if (method == "matrix") {
    out[py::str(prefix + "matrix")] =
        extract_matrix(profile, pearson, window, height, width);
  } else {  // knn
    auto t = extract_knn(profile, pearson, window);
    out[py::str(prefix + "match_cols")] = std::move(std::get<0>(t));
    out[py::str(prefix + "match_rows")] = std::move(std::get<1>(t));
    out[py::str(prefix + "match_dist")] = std::move(std::get<2>(t));
  }
}

py::dict do_join(const std::vector<double>& a, py::object b_obj, int m,
                 const std::string& method, bool want_index, bool left_right,
                 int k, const py::kwargs& kwargs) {
  const bool self_join = b_obj.is_none();

  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.window = m;
  if (self_join) {
    args.timeseries_b = a;
    args.has_b = false;
  } else {
    auto barr =
        b_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    args.timeseries_b = ArrayToDoubleVector(barr, "b");
    args.has_b = true;
  }

  args.profile_type = ProfileTypeForMethod(method, want_index);
  args.profile_a.type = args.profile_type;
  args.profile_b.type = args.profile_type;
  if (method == "knn") {
    args.max_matches_per_column = k;
  }

  // Direction control. A self-join always computes the upper triangle (both
  // row and column reductions); keep_rows_separate decides whether the row
  // direction is merged into the combined profile (left_right=False) or kept
  // separate as profile_b (left_right=True). An ab-join only computes the row
  // direction when the caller wants it.
  args.computing_columns = true;
  args.computing_rows = self_join ? true : left_right;
  args.keep_rows_separate = left_right;

  bool pearson = setup_and_do_SCAMP(&args, kwargs);

  py::dict out;
  if (left_right) {
    PutDirection(out, "left_", method, want_index, pearson, m,
                 args.matrix_height, args.matrix_width, args.profile_a);
    PutDirection(out, "right_", method, want_index, pearson, m,
                 args.matrix_height, args.matrix_width, args.profile_b);
  } else {
    PutDirection(out, "", method, want_index, pearson, m, args.matrix_height,
                 args.matrix_width, args.profile_a);
  }
  return out;
}

bool has_gpu_support() { return SCAMP::num_available_gpus() > 0; }

// Runs the SCAMP GPU autotuner over the requested device(s) and writes the
// chosen kernel configuration to the on-disk cache. Returns the number of
// devices tuned. Raises ImportError-equivalent (RuntimeError) when pyscamp
// was built without CUDA, and a ValueError when no GPUs are available.
int run_autotune(const std::vector<int>& devices,
                 const std::string& cache_path) {
#ifndef _HAS_CUDA_
  (void)devices;
  (void)cache_path;
  throw std::runtime_error(
      "pyscamp was built without CUDA; autotune() is unavailable.");
#else
  std::vector<int> targets = devices;
  if (targets.empty()) {
    int num_dev = 0;
    cudaGetDeviceCount(&num_dev);
    if (num_dev <= 0) {
      throw std::invalid_argument(
          "No CUDA devices available; pyscamp.autotune() needs at least one.");
    }
    // Default to device 0 only -- a full sweep takes O(minutes) and the
    // typical multi-GPU box has identical devices, so tuning them all
    // burns wall time on identical configs. Callers who really do want
    // to tune multiple distinct GPUs should pass devices=[0, 1, ...]
    // explicitly. Mirrors the CLI --autotune behavior in main.cpp.
    if (num_dev > 1) {
      py::print("pyscamp.autotune():", num_dev,
                "GPUs visible; tuning device 0 only (pass devices=[...]"
                " to override).");
    }
    targets.push_back(0);
  }
  // Shares its bench impl + verbose progress format with the CLI's
  // --autotune path (which calls RunAutotuneWithBenchmark directly
  // from main.cpp).
  for (int dev : targets) {
    SCAMP::RunAutotuneWithBenchmark(dev, &SCAMP::DefaultBenchmarkVariant,
                                    cache_path, /*verbose=*/true);
  }
  return static_cast<int>(targets.size());
#endif
}

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

// Compiled core of the pyscamp package. Imported as pyscamp._core; the
// pyscamp/__init__.py re-exports the public names and adds the pure-Python
// join() surface. (Before 5.0 this module was imported directly as `pyscamp`.)
PYBIND11_MODULE(_core, m) {
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

  // Low-level unified join backing pyscamp.join(). Structural arguments are
  // explicit; everything else flows through kwargs. Returns a dict whose keys
  // (profile / index / matrix / match_cols|rows|dist, each optionally with a
  // left_/right_ prefix) are assembled into a JoinResult by the Python layer.
  // Underscore-prefixed: not part of the public surface -- use join().
  m.def(
      "_do_join",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
         py::object b, int m, const std::string& method, bool want_index,
         bool left_right, int k, const py::kwargs& kwargs) {
        return do_join(ArrayToDoubleVector(a, "a"), b, m, method, want_index,
                       left_right, k, kwargs);
      },
      py::arg("a"), py::arg("b"), py::arg("m"), py::arg("method"),
      py::arg("want_index"), py::arg("left_right"), py::arg("k"));

  m.def("autotune", &run_autotune, py::arg("devices") = std::vector<int>{},
        py::arg("cache_path") = std::string{}, R"pbdoc(
    Run the SCAMP GPU kernel autotuner for the selected device(s) and persist
    the chosen kernel configurations to disk. Future pyscamp calls on the same
    machine will read these configurations from the cache and use them when
    launching GPU kernels.

    A full sweep takes a few minutes on a recent GPU. The output is verbose
    so you can follow progress; pass ``cache_path`` to redirect the write
    elsewhere (e.g. for a sandboxed run).

    :param devices: List of CUDA device IDs to tune. If empty (default),
                    only device 0 is tuned -- a full sweep takes O(minutes)
                    and most multi-GPU boxes hold identical devices, so
                    sweeping them all wastes wall time on identical configs.
                    Pass ``devices=[0, 1, ...]`` explicitly to tune more
                    than one (e.g. if you have two different GPU models).
    :type devices: list[int], optional
    :param cache_path: Filesystem path to read/write the cache from. Empty
                       (default) resolves in this order: ``$SCAMP_AUTOTUNE_CACHE``
                       (if set, used verbatim), then ``$XDG_CACHE_HOME/scamp/autotune.txt``
                       (if set), then a platform-specific user dir
                       (``$HOME/.cache/scamp/autotune.txt`` on Linux/macOS;
                       ``%LOCALAPPDATA%\scamp\autotune.txt`` on Windows).
                       Parent directories are created automatically.
    :type cache_path: str, optional
    :return: Number of devices that were tuned.
    :rtype: int
    :raises RuntimeError: If pyscamp was built without CUDA support.
    :raises ValueError: If no CUDA devices are available.
    )pbdoc");

  m.def("selfjoin",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           int m, const py::kwargs& kwargs) {
          return self_join_1NN_INDEX(ArrayToDoubleVector(a, "a"), m, kwargs);
        },
        py::arg("a"), py::arg("m"), R"pbdoc(
    Computes the matrix profile for time series A.
  
    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :return: A tuple containing the matrix profile as the first element and the indices as a the second element.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    )pbdoc");

  m.def("abjoin",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int m, const py::kwargs& kwargs) {
          return ab_join_1NN_INDEX(ArrayToDoubleVector(a, "a"),
                                   ArrayToDoubleVector(b, "b"), m, kwargs);
        },
        py::arg("a"), py::arg("b"), py::arg("m"),
        R"pbdoc(
    For each subsequence in time series A, finds the nearest neighbor in time series B.

    :param a: Time series, b will be queried for subsequences in a.
    :type a: 1D array
    :param b: Time series in which to search for matches for subsequences in a.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param allow_trivial_match: When True (default), all subsequence pairs are considered. When False, treats a and b as aligned (e.g. overlapping segments of the same series) and excludes trivial self-matches near the equivalent main diagonal.
    :type allow_trivial_match: bool, optional
    :return: A tuple. First element: The nearest neighbor distance of subsequences in a to time series b. Second element: The index (in b) of each nearest neighbor.
    :rtype: Tuple of np.ndarray[float32] and np.ndarray[int32]
    )pbdoc");

  m.def("selfjoin_sum",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           int m, const py::kwargs& kwargs) {
          return self_join_SUM_THRESH(ArrayToDoubleVector(a, "a"), m, kwargs);
        },
        py::arg("a"), py::arg("m"),
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

  m.def("abjoin_sum",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int m, const py::kwargs& kwargs) {
          return ab_join_SUM_THRESH(ArrayToDoubleVector(a, "a"),
                                    ArrayToDoubleVector(b, "b"), m, kwargs);
        },
        py::arg("a"), py::arg("b"), py::arg("m"), R"pbdoc(
    For each subsequence in time series a, returns the sum of the correlations to subsequences in time series b above specified threshold (default 0).

    :param a: Time series to compute matrix profile for.
    :type a: 1D array
    :param b: Time series to search for matches.
    :type b: 1D array
    :param m: Subsequence length to use for computing the matrix profile.
    :type m: int
    :param threshold: Correlation threshold [0,1] (Default 0), matches which have a correlation less than the threshold will be ignored
    :type threshold: float, optional
    :param allow_trivial_match: When True (default), all subsequence pairs are considered. When False, treats a and b as aligned (e.g. overlapping segments of the same series) and excludes trivial self-matches near the equivalent main diagonal.
    :type allow_trivial_match: bool, optional
    :return: For each subsequence in A, returns the sum of correlations above the the specified threshold in B.
    :rtype: np.ndarray[float64]
    )pbdoc");

  m.def("selfjoin_knn",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           int m, int k, const py::kwargs& kwargs) {
          return self_join_KNN(ArrayToDoubleVector(a, "a"), m, k, kwargs);
        },
        py::arg("a"), py::arg("m"), py::arg("k"),
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

  m.def("abjoin_knn",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int m, int k, const py::kwargs& kwargs) {
          return ab_join_KNN(ArrayToDoubleVector(a, "a"),
                             ArrayToDoubleVector(b, "b"), m, k, kwargs);
        },
        py::arg("a"), py::arg("b"), py::arg("m"), py::arg("k"), R"pbdoc(
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
    :param allow_trivial_match: When True (default), all subsequence pairs are considered. When False, treats a and b as aligned (e.g. overlapping segments of the same series) and excludes trivial self-matches near the equivalent main diagonal.
    :type allow_trivial_match: bool, optional
    :return: List of tuples (col, row, distance) containing the matches (up to K) for each column of the distance matrix, col is the index in A, row is the index in B of the match, and d is the distance between the two subsequences
    :rtype: List of tuple[int, int, float]
    )pbdoc");

  m.def("selfjoin_matrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           int m, const py::kwargs& kwargs) {
          return self_join_MATRIX(ArrayToDoubleVector(a, "a"), m, kwargs);
        },
        py::arg("a"), py::arg("m"),
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

  m.def("abjoin_matrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int m, const py::kwargs& kwargs) {
          return ab_join_MATRIX(ArrayToDoubleVector(a, "a"),
                                ArrayToDoubleVector(b, "b"), m, kwargs);
        },
        py::arg("a"), py::arg("b"), py::arg("m"), R"pbdoc(
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
    :param allow_trivial_match: When True (default), all subsequence pairs are considered. When False, treats a and b as aligned (e.g. overlapping segments of the same series) and excludes trivial self-matches near the equivalent main diagonal.
    :type allow_trivial_match: bool, optional
    :return: A 2D array of height of mheight and width of mwidth. This is a pooled version of the full distance matrix.
    :rtype: 2D array
)pbdoc");

  m.attr("__version__") = "dev";
}
