include(SCAMPMacros)

add_library(dispatch_baseline  ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_baseline.cpp)
add_library(kernel_baseline ${CMAKE_SOURCE_DIR}/src/core/cpu_kernel/cpu_kernels.cpp)

if (SCAMP_USE_PERFORMANCE_OPTIMIZATION_FLAGS)
  enable_compiler_performance_checks()
endif()

target_link_libraries(kernel_baseline common kernel_common Eigen3::Eigen)
target_link_libraries(dispatch_baseline kernel_baseline)

mark_clang_tidy()
