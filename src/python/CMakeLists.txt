add_library(pyscamp MODULE SCAMP_python.cpp)
set_target_properties(pyscamp PROPERTIES 
  INTERPROCEDURAL_OPTIMIZATION TRUE
  CXX_VISIBILITY_PRESET "hidden"
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(pyscamp PRIVATE scamp_args scamp_utils scamp_interface common)
target_link_libraries(pyscamp PRIVATE pybind11::module)

set(CURR_LIBS pyscamp)

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
