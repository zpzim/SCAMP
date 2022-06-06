include(SCAMPMacros)

CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_OPT_AVX_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mfma" COMPILER_OPT_FMA_SUPPORTED)

add_library(dispatch_avx  ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_avx.cpp)
add_library(kernel_avx ${CMAKE_SOURCE_DIR}/src/core/cpu_kernel/cpu_kernels.cpp)


target_compile_definitions(kernel_avx PRIVATE "-D_SCAMP_USE_AVX_")
target_compile_definitions(dispatch_avx PRIVATE "-D_SCAMP_USE_AVX_")

if (COMPILER_OPT_AVX_SUPPORTED)
  target_compile_options(kernel_avx PRIVATE "-mavx")
  target_compile_options(dispatch_avx PRIVATE "-mavx")
endif()

if (COMPILER_OPT_ARCH_AVX_SUPPORTED)
  target_compile_options(kernel_avx PRIVATE "/arch:AVX")
  target_compile_options(dispatch_avx PRIVATE "/arch:AVX")
endif()

if (SCAMP_USE_PERFORMANCE_OPTIMIZATION_FLAGS)
  enable_compiler_performance_checks()
endif()

target_link_libraries(kernel_avx common kernel_common Eigen3::Eigen)
target_link_libraries(dispatch_avx kernel_avx)

mark_clang_tidy()
