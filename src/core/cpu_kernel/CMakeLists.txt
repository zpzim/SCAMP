include(SCAMPMacros)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baseline)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx2)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dispatcher.cpp)
target_link_libraries(cpu_kernels dispatch_baseline dispatch_avx dispatch_avx2 CpuFeature::cpu_features)

mark_clang_tidy()
