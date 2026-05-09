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

  # Maxwell (SM 5.x): deprecated CUDA 12.0, removed CUDA 13.0
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 50 52 53)
  endif()

  # Pascal (SM 6.x): removed in CUDA 13.0
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60 61 62)
  endif()

  # Volta (SM 7.0, 7.2): introduced CUDA 9.0/10.0; removed CUDA 13.0
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70 72)
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

  # Blackwell datacenter (SM 100, 101) + consumer (SM 120): introduced CUDA 12.8
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 101 120)
  endif()

  # Blackwell SM 103 + SM 121: introduced CUDA 12.9
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 103 121)
  endif()

  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
  list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL)
  message(STATUS "Configuring CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endmacro()
