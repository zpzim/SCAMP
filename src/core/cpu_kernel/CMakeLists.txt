include(SCAMPMacros)

# Force BUILD_TESTING OFF so that cpu_features does not try to download
# googletest. gRPC (pulled in via BUILD_CLIENT_SERVER) sets BUILD_TESTING=ON
# in the CMake CACHE before this directory is processed; FORCE overrides it.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

fetch_env(SCAMP_USE_EXTERNAL_EIGEN)

if (SCAMP_USE_EXTERNAL_EIGEN)
  find_package(Eigen3 5.0.0 REQUIRED NO_MODULE)
else()
  # Create a header-only interface target instead of using Eigen's CMakeLists.txt.
  # Eigen's blas/CMakeLists.txt unconditionally calls check_language(Fortran),
  # which stalls cmake on Windows (90+ minutes) when no Fortran compiler exists.
  # SCAMP only uses Eigen as a header-only SIMD library so no compiled targets
  # from Eigen are needed.
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/eigen")
  endif()
endif()

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cpu_features ${CMAKE_CURRENT_BINARY_DIR}/cpu_features)
  add_definitions("-D_SCAMP_DISTRIBUTABLE_")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|X86|amd64|AMD64|i386|i686|x86_64)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/avx2)
  endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/baseline)

add_library(cpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dispatcher.cpp)

if (SCAMP_ENABLE_BINARY_DISTRIBUTION)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|X86|amd64|AMD64|i386|i686|x86_64)")
    target_link_libraries(cpu_kernels dispatch_baseline dispatch_avx dispatch_avx2 CpuFeatures::cpu_features)
  else()
    target_link_libraries(cpu_kernels dispatch_baseline CpuFeatures::cpu_features)
  endif()
else()
  target_link_libraries(cpu_kernels dispatch_baseline)
endif()

mark_clang_tidy()
