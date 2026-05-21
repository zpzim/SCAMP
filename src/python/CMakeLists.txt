include(SCAMPMacros)

fetch_env(PYSCAMP_USE_EXTERNAL_PYBIND11)

set(PYBIND11_FIND_PYTHON ON)

if (PYSCAMP_USE_EXTERNAL_PYBIND11)
  find_package(pybind11 2.13.6 CONFIG REQUIRED)
else()
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
endif()

add_library(pyscamp MODULE SCAMP_python.cpp)
set_target_properties(pyscamp PROPERTIES 
  INTERPROCEDURAL_OPTIMIZATION TRUE
  CXX_VISIBILITY_PRESET "hidden"
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(pyscamp PRIVATE scamp_args scamp_utils scamp_interface common)
target_link_libraries(pyscamp PRIVATE pybind11::module)

if (CMAKE_CUDA_COMPILER)
  # pyscamp.autotune() routes to SCAMP::RunAutotuneWithBenchmark, which
  # lives in autotune_bench (depends on scamp_op via do_SCAMP for each
  # timed trial). autotune_bench's own deps -- gpu_utils, scamp_op,
  # scamp_args, common -- come along transitively, so we only need to
  # add autotune_bench + CUDA::cudart here.
  target_link_libraries(pyscamp PRIVATE autotune_bench gpu_utils CUDA::cudart)
endif()

mark_cuda_if_available()
mark_clang_tidy()
