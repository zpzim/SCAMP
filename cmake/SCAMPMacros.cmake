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
  if (CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_90,code=sm_90")
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_86,code=sm_86")
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 80)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_80,code=sm_80")
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_75,code=sm_75")
  endif()
  list(APPEND CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72)
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_70")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_61,code=sm_61")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_62,code=sm_62")
  set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_60,code=sm_60")
  if (CUDA_VERSION VERSION_LESS "12.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_50,code=sm_50")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_52,code=sm_52")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_53,code=sm_53")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_37,code=sm_37")
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_35,code=sm_35")
  endif()
  if (CUDA_VERSION VERSION_LESS "11.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 30)
    set(CUDA_GENCODE_FLAGS "${CUDA_GENCODE_FLAGS} -gencode arch=compute_30,code=sm_30")
  endif()

  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
  list(SORT CMAKE_CUDA_ARCHITECTURES)
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    message(STATUS "Configuring CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    unset(CUDA_GENCODE_FLAGS)
  else()
    unset(CMAKE_CUDA_ARCHITECTURES)
  endif()
endmacro() 
