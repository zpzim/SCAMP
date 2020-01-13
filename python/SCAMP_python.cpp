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
  args.max_matches_per_column = 10;
  args.matrix_height = -1.0;
  args.matrix_width = -1.0;
  args.profile_a.type = profile_type;
  args.profile_b.type = profile_type;
  args.profile_a.matrix_height = -1.0;
  args.profile_b.matrix_height = -1.0;
  args.profile_a.matrix_width = -1.0;
  args.profile_b.matrix_width = -1.0;
  args.profile_a.output_matrix = false;
  args.profile_b.output_matrix = false;
  return args;
}

std::tuple<std::vector<float>, std::vector<int>> SCAMP_AB(
    const std::vector<double>& a, const std::vector<double>& b, int m) {
  SCAMP::SCAMPArgs args = GetDefaultSCAMPArgs();
  args.timeseries_a = a;
  args.timeseries_b = b;
  args.window = m;
  args.has_b = true;
  args.max_tile_size = 128000;
  args.computing_rows = false;
  args.computing_columns = true;

  InitProfileMemory(&args);
  SCAMP::do_SCAMP(&args);
  std::vector<float> NN;
  std::vector<int> index;
  SplitProfile1NNINDEX(args.profile_a.data[0].uint64_value, &NN, &index);
  return std::make_tuple(NN, index);
}

std::tuple<std::vector<float>, std::vector<int>> SCAMP_SELF(
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

namespace py = pybind11;

PYBIND11_MODULE(pySCAMP, m) {
  m.doc() = R"pbdoc(
        SCAMP: SCAlable Matrix Profile
        -------------------------------
        .. currentmodule:: scamp
        .. autosummary::
           :toctree: _generate
           SCAMP_AB
           SCAMP_SELF
    )pbdoc";

  m.def("SCAMP_AB", &SCAMP_AB, R"pbdoc(
        Returns the ab-join matrix profile of 2 time series (in Pearson Correlation)
    )pbdoc");

  m.def("SCAMP_SELF", &SCAMP_SELF, R"pbdoc(
        Returns the self-join of 2 time series (in Pearson Correlation)
  )pbdoc");

  m.attr("__version__") = "dev";
}
