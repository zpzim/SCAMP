include(SCAMPMacros)

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  add_definitions("-D_SCAMP_DISTRIBUTABLE_")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx2)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baseline)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dispatcher.cpp)

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  target_link_libraries(cpu_kernels dispatch_baseline dispatch_avx dispatch_avx2 CpuFeature::cpu_features)
else()
  target_link_libraries(cpu_kernels dispatch_baseline CpuFeature::cpu_features)
endif()

mark_clang_tidy()
