add_library(common ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

if (CMAKE_CUDA_COMPILER) 
  target_link_libraries(common CUDA::cudart)
endif()

add_library(profile ${CMAKE_CURRENT_SOURCE_DIR}/profile.cpp)
target_link_libraries(profile common)

add_library(scamp_args ${CMAKE_CURRENT_SOURCE_DIR}/scamp_args.cpp)
target_link_libraries(scamp_args profile common)

add_library(scamp_interface ${CMAKE_CURRENT_SOURCE_DIR}/scamp_interface.cpp)
target_link_libraries(scamp_interface scamp_args scamp_op)

add_library(scamp_utils ${CMAKE_CURRENT_SOURCE_DIR}/scamp_utils.cpp)
target_link_libraries(scamp_utils common profile)

mark_cuda_if_available()
mark_clang_tidy()
