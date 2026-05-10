macro(mark_clang_tidy)
  get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  if(CLANG_TIDY_EXE)
    foreach(lib ${current_targets})
      set_target_properties(
        "${lib}" PROPERTIES
        CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
      )
    endforeach(lib)
  endif()
endmacro()

macro(mark_cuda_if_available)
  get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  if (CMAKE_CUDA_COMPILER)
    foreach(lib ${current_targets})
      target_compile_definitions("${lib}" PUBLIC -D_HAS_CUDA_)
    endforeach(lib)
  endif()
endmacro()

macro(enable_compiler_performance_checks)
  # Checks for vectorization and performance analysis on the CPU kernels.
  # Turn these on only as needed since they cause a lot of unnecessary compiler output.
  CHECK_CXX_COMPILER_FLAG("-Winline" COMPILER_OPT_WARN_INLINE_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("/Qvec-report:2" COMPILER_OPT_QVEC_REPORT_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("-fopt-info-vec-all" COMPILER_OPT_GCC_VEC_INFO_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("-Rpass-analysis=loop-vectorize" COMPILER_OPT_LLVM_VEC_MISSED_INFO_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("-Rpass=loop-vectorize" COMPILER_OPT_LLVM_VEC_LOOPS_INFO_SUPPORTED)

  if (COMPILER_OPT_QVEC_REPORT_SUPPORTED)
    add_compile_options("/Qvec-report:2")
  endif()

  if (COMPILER_OPT_GCC_VEC_INFO_SUPPORTED)
    add_compile_options("-fopt-info-vec-all")
  endif()

  if (COMPILER_OPT_LLVM_VEC_MISSED_INFO_SUPPORTED)
    add_compile_options("-Rpass-analysis=loop-vectorize")
  endif()

  if (COMPILER_OPT_LLVM_VEC_LOOPS_INFO_SUPPORTED)
    add_compile_options("-Rpass=loop-vectorize")
  endif()

  if (COMPILER_OPT_WARN_INLINE_SUPPORTED)
    add_compile_options("-Winline")
  endif()
endmacro()

macro(fetch_env ENVVAR)
  if (DEFINED ENV{${ENVVAR}})
    set(${ENVVAR} "$ENV{${ENVVAR}}")
  endif()
endmacro()

macro(set_cuda_architectures)
  message(STATUS "CUDA VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

  # Rebuild from scratch; the placeholder set before enable_language(CUDA) is replaced here.
  set(CMAKE_CUDA_ARCHITECTURES "")

  # Kepler (SM 3.x): removed in CUDA 12.0
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 35 37)
  endif()

  # Maxwell (SM 5.x): deprecated CUDA 12.0, removed CUDA 13.0.
  # Excluded from CUDA 12.8+ builds: CCCL 2.8.x (CUDA 12.9.x) has a
  # preprocessor macro arity bug (NVIDIA/cccl#4967) — the namespace-name
  # token concatenation overflows when too many arch numbers are included.
  # Dropping Maxwell+Volta (and minor Blackwell variants below) on 12.8+
  # keeps the list within the limit. Fixed in CCCL 3.0.0 / CUDA 13.0.
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 50 52 53)
  endif()

  # Pascal (SM 6.0/6.1) + Volta (SM 7.0): removed in CUDA 13.0.
  # SM 6.2 (Tegra X2, embedded only) and SM 7.2 (Jetson Xavier, embedded
  # only) are excluded from 12.8+ builds to help stay within the CCCL
  # 2.8.x arch-token limit (see Maxwell note above).
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60 61 70)
  endif()

  # SM 6.2 (Tegra X2) + SM 7.2 (Jetson Xavier): embedded targets excluded
  # from CUDA 12.8+ to reduce arch count (see Maxwell note above)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 62 72)
  endif()

  # Turing (SM 7.5): introduced CUDA 10.0
  list(APPEND CMAKE_CUDA_ARCHITECTURES 75)

  # Ampere A100/A30 (SM 8.0): introduced CUDA 11.0 (our minimum)
  list(APPEND CMAKE_CUDA_ARCHITECTURES 80)

  # Ampere desktop/server (SM 8.6): introduced CUDA 11.1
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  # Ampere Jetson Orin (SM 8.7): introduced CUDA 11.4
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.4")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 87)
  endif()

  # Ada Lovelace (SM 8.9) + Hopper (SM 9.0): introduced CUDA 11.8
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 89 90)
  endif()

  # Blackwell datacenter (SM 100) + consumer (SM 120): introduced CUDA 12.8.
  # SM 101, 103, 121 (minor variants) are omitted on CUDA 12.x to stay within
  # the CCCL 2.8.x arch-token limit (NVIDIA/cccl#4967); they're included on
  # CUDA 13.0+ where the bug is fixed.
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8" AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
  endif()

  # Blackwell all variants on CUDA 13.0+ (CCCL 3.0.0 fixes the arity bug).
  # sm_101 (Jetson Thor) was renamed to sm_110 in CUDA 13.0; sm_103 (B300)
  # and sm_121 (DGX Spark) were introduced in CUDA 12.9 but kept out of 12.x
  # builds to stay within the CCCL 2.8.x arch-token limit.
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 103 110 120 121)
  endif()

  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
  list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL)
  message(STATUS "Configuring CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endmacro()
