include(SCAMPMacros)

set(BUILD_TESTING OFF)

fetch_env(SCAMP_USE_EXTERNAL_EIGEN)

if (SCAMP_USE_EXTERNAL_EIGEN)
  find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
else()
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/eigen ${CMAKE_CURRENT_BINARY_DIR}/eigen)
endif()

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cpu_features ${CMAKE_CURRENT_BINARY_DIR}/cpu_features)
  add_definitions("-D_SCAMP_DISTRIBUTABLE_")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx2)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baseline)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dispatcher.cpp)

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  target_link_libraries(cpu_kernels dispatch_baseline dispatch_avx dispatch_avx2 CpuFeature::cpu_features)
else()
  target_link_libraries(cpu_kernels dispatch_baseline)
endif()

mark_clang_tidy()
