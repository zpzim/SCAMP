add_library(qt_helper ${CMAKE_CURRENT_SOURCE_DIR}/qt_helper.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)
  target_link_libraries(qt_helper ${CUDA_CUFFT_LIBRARIES} qt_kernels common)
else()
  target_link_libraries(qt_helper common)
endif()

add_library(kernel_common ${CMAKE_CURRENT_SOURCE_DIR}/kernel_common.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
  target_link_libraries(gpu_utils kernel_common ${CUDA_cudart_LIBRARY})
  add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu)
  target_link_libraries(gpu_kernels gpu_utils kernel_common common)
endif()

add_library(cpu_stats ${CMAKE_CURRENT_SOURCE_DIR}/cpu_stats.cpp)
target_link_libraries(cpu_stats common)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernels.cpp)
target_link_libraries(cpu_kernels kernel_common common)

CHECK_CXX_COMPILER_FLAG("-Winline" COMPILER_OPT_WARN_INLINE_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("/Qvec-report:2" COMPILER_OPT_QVEC_REPORT_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-fopt-info-vec-all" COMPILER_OPT_GCC_VEC_INFO_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-Rpass-analysis=loop-vectorize" COMPILER_OPT_LLVM_VEC_MISSED_INFO_SUPPORTED)
CHECK_CXX_COMPILER_FLAG("-Rpass=loop-vectorize" COMPILER_OPT_LLVM_VEC_LOOPS_INFO_SUPPORTED)


if (COMPILER_OPT_QVEC_REPORT_SUPPORTED)
  target_compile_options(cpu_kernels PRIVATE "/Qvec-report:2")
endif()

if (COMPILER_OPT_GCC_VEC_INFO_SUPPORTED)
  target_compile_options(cpu_kernels PRIVATE "-fopt-info-vec-all")
endif()

if (COMPILER_OPT_LLVM_VEC_MISSED_INFO_SUPPORTED)
  target_compile_options(cpu_kernels PRIVATE "-Rpass-analysis=loop-vectorize")
endif()

if (COMPILER_OPT_LLVM_VEC_LOOPS_INFO_SUPPORTED)
  target_compile_options(cpu_kernels PRIVATE "-Rpass=loop-vectorize")
endif()


if (COMPILER_OPT_WARN_INLINE_SUPPORTED)
  target_compile_options(cpu_kernels PRIVATE "-Winline")
endif()

add_library(tile ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp)

if (CMAKE_CUDA_COMPILER)
  target_link_libraries(tile common gpu_kernels cpu_kernels profile qt_helper)
else()
  target_link_libraries(tile common cpu_kernels profile qt_helper)
endif()

add_library(scamp_op ${CMAKE_CURRENT_SOURCE_DIR}/SCAMP.cpp)
target_link_libraries(scamp_op Threads::Threads tile cpu_stats common profile qt_helper)

# Set the correct compile/gencode flags for CUDA libraries.
if (CMAKE_CUDA_COMPILER)
  set_target_properties(gpu_kernels PROPERTIES
    COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}"
  )
  set_target_properties(qt_kernels PROPERTIES
    COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}"
  )
  set_target_properties(gpu_utils PROPERTIES
    COMPILE_FLAGS "${COMPILE_FLAGS} ${CUDA_GENCODE_FLAGS}"
  )
endif()

set(CURR_LIBS common profile scamp_args scamp_interface scamp_utils)

if (CMAKE_CUDA_COMPILER)
  set(CURR_LIBS ${CURR_LIBS} qt_kernels gpu_utils gpu_kernels)
endif()

if (CMAKE_CUDA_COMPILER)
  foreach(lib ${CURR_LIBS})
    target_compile_definitions("${lib}" PUBLIC -D_HAS_CUDA_)
  endforeach(lib)
endif()

# Add clang tidy rules
if(CLANG_TIDY_EXE)
  foreach(lib ${MAIN_LIBS})
    set_target_properties(
      "${lib}" PROPERTIES
      CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
    )
  endforeach(lib)
endif()
