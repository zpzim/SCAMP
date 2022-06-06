include(SCAMPMacros)

# Build targets for computing Sliding Dot Product.
add_library(qt_helper ${CMAKE_CURRENT_SOURCE_DIR}/qt_helper.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)
  target_link_libraries(qt_helper ${CUDA_CUFFT_LIBRARIES} qt_kernels common)
else()
  target_link_libraries(qt_helper common)
endif()

# Build SCAMP GPU/CPU kernel targets.
add_library(kernel_common ${CMAKE_CURRENT_SOURCE_DIR}/kernel_common.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
  target_link_libraries(gpu_utils kernel_common ${CUDA_cudart_LIBRARY})
  add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu)
  target_link_libraries(gpu_kernels gpu_utils kernel_common common)
endif()

add_library(cpu_stats ${CMAKE_CURRENT_SOURCE_DIR}/cpu_stats.cpp)
target_link_libraries(cpu_stats common)

add_library(tile ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp)

# Defines 'cpu_kernels' target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernel)

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

mark_cuda_if_available()
mark_clang_tidy()
