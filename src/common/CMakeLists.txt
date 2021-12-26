add_library(common ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp)

if (CMAKE_CUDA_COMPILER) 
  target_link_libraries(common ${CUDA_cudart_LIBRARY})
endif()

add_library(profile ${CMAKE_CURRENT_SOURCE_DIR}/profile.cpp)
target_link_libraries(profile common)

add_library(scamp_args ${CMAKE_CURRENT_SOURCE_DIR}/scamp_args.cpp)
target_link_libraries(scamp_args profile common)

# Depends on scamp_op in src/common 
add_library(scamp_interface ${CMAKE_CURRENT_SOURCE_DIR}/scamp_interface.cpp)
target_link_libraries(scamp_interface scamp_args scamp_op)

add_library(scamp_utils ${CMAKE_CURRENT_SOURCE_DIR}/scamp_utils.cpp)
target_link_libraries(scamp_utils common profile)

set(CURR_LIBS common profile scamp_args scamp_interface scamp_utils)

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
