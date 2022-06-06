include(SCAMPMacros)

CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVX2_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_OPT_AVX_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_OPT_AVX2_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-mfma" COMPILER_OPT_FMA_SUPPORTED)

add_library(kernel_avx2 ${CMAKE_SOURCE_DIR}/src/core/cpu_kernel/cpu_kernels.cpp)
add_library(dispatch_avx2 ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_avx2.cpp)
  
target_compile_definitions(kernel_avx2 PRIVATE "_SCAMP_USE_AVX2_")
target_compile_definitions(dispatch_avx2 PRIVATE "_SCAMP_USE_AVX2_")

if (COMPILER_OPT_AVX_SUPPORTED AND COMPILER_OPT_AVX2_SUPPORTED AND COMPILER_OPT_FMA_SUPPORTED)
  target_compile_options(kernel_avx2 PRIVATE "-mavx" PRIVATE "-mavx2" PRIVATE "-mfma")
  target_compile_options(dispatch_avx2 PRIVATE "-mavx" PRIVATE "-mavx2" PRIVATE "-mfma")
endif()

if (COMPILER_OPT_ARCH_AVX_SUPPORTED AND COMPILER_OPT_ARCH_AVX2_SUPPORTED)
  target_compile_options(kernel_avx2 PRIVATE "/arch:AVX" PRIVATE "/arch:AVX2")
  target_compile_options(dispatch_avx2 PRIVATE "/arch:AVX" PRIVATE "/arch:AVX2")
endif()

if (SCAMP_USE_PERFORMANCE_OPTIMIZATION_FLAGS)
  enable_compiler_performance_checks()
endif()

target_link_libraries(kernel_avx2 common kernel_common Eigen3::Eigen)
target_link_libraries(dispatch_avx2 kernel_avx2)

mark_clang_tidy()
