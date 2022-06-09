include(SCAMPMacros)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/grpc ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)

# After using add_subdirectory, we can now use the grpc targets directly from
# this build.
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_GRPC grpc)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Add include directories we will use later
include_directories(${CMAKE_SOURCE_DIR}/third_party/grpc/include/) 
include_directories(${CMAKE_SOURCE_DIR}/third_party/grpc/third_party/gflags/include/)
include_directories(${CMAKE_SOURCE_DIR}/third_party/grpc/third_party/protobuf/src/)
include_directories(${CMAKE_SOURCE_DIR}/third_party/grpc/third_party/abseil-cpp/)

get_filename_component(hw_proto "${CMAKE_CURRENT_SOURCE_DIR}/scamp.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/scamp.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scamp.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/scamp.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scamp.grpc.pb.h")

# Compile Protos
add_custom_command(
  OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS -I "${hw_proto_path}"
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}/"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}/"
  --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE} ${hw_proto}
  DEPENDS "${hw_proto}")

message(STATUS "Using grpc lib ${_GRPC_CPP_UNSECURE}") 
message(STATUS "Using grpc cpp plugin ${_GRPC_CPP_PLUGIN_EXECUTABLE}") 
message(STATUS "Using proto lib ${_PROTOBUF_LIBPROTOBUF}")

# Proto Messages Lib
add_library(messages ${hw_proto_srcs} ${hw_proto_headers} ${hw_grpc_srcs} ${hw_grpc_headers})
target_link_libraries(messages ${_PROTOBUF_LIBPROTOBUF} ${_GRPC_GRPCPP} ${_GRPCPP_UNSECURE} -lz -lpthread)

# Add generated proto headers to includes
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(kube_utils ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)
target_link_libraries(kube_utils scamp_interface messages)

add_library(distributed_tile ${CMAKE_CURRENT_SOURCE_DIR}/distributed_tile.cpp)
target_link_libraries(distributed_tile messages)

add_library(distributed_job ${CMAKE_CURRENT_SOURCE_DIR}/distributed_job.cpp)
target_link_libraries(distributed_job distributed_tile messages)

add_library(job_list ${CMAKE_CURRENT_SOURCE_DIR}/job_list.cpp)
target_link_libraries(job_list distributed_job messages)

add_library(worker ${CMAKE_CURRENT_SOURCE_DIR}/scamp_worker.cpp)
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(worker -lcudart_static common scamp_interface kube_utils messages )
else()
  target_link_libraries(worker common scamp_interface kube_utils messages )
endif()

add_library(scamp_distributed_interface ${CMAKE_CURRENT_SOURCE_DIR}/scamp_interface.cpp)
target_link_libraries(scamp_distributed_interface kube_utils messages )

add_executable(SCAMPclient ${CMAKE_CURRENT_SOURCE_DIR}/scamp_client.cc)
add_executable(SCAMPserver ${CMAKE_CURRENT_SOURCE_DIR}/scamp_server.cc)
target_link_libraries(SCAMPserver kube_utils job_list distributed_job messages -Wl,--whole-archive grpc++_reflection -Wl,--no-whole-archive)
target_link_libraries(SCAMPclient worker)

add_definitions("-D_DISTRIBUTED_EXECUTION_")

add_executable(SCAMP_distributed ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(SCAMP_distributed -lcudart_static gflags common scamp_utils scamp_distributed_interface)
else()
  target_link_libraries(SCAMP_distributed gflags common scamp_distributed_interface scamp_utils)
endif() 

mark_cuda_if_available()
mark_clang_tidy()
