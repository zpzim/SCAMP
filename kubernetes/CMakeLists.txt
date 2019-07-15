
get_filename_component(hw_proto "${CMAKE_CURRENT_SOURCE_DIR}/scamp.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/scamp.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scamp.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/scamp.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scamp.grpc.pb.h")

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
add_library(messages ${hw_proto_srcs} ${hw_proto_headers} ${hw_grpc_srcs} ${hw_grpc_headers})
target_link_libraries(messages ${_PROTOBUF_LIBPROTOBUF} ${_GRPC_GRPCPP} ${_GRPC_CPP_UNSECURE} -lz -lpthread)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
add_executable(SCAMPclient ${CMAKE_CURRENT_SOURCE_DIR}/scamp_client.cc)
add_executable(SCAMPserver ${CMAKE_CURRENT_SOURCE_DIR}/scamp_server.cc)
target_link_libraries(SCAMPserver messages)
if (CMAKE_CUDA_COMPILER)
    target_link_libraries(SCAMPclient -lcudart_static common scamp messages )
else()
    target_link_libraries(SCAMPclient common scamp messages )
endif()

