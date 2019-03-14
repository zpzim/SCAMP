find_package( Protobuf REQUIRED ) 

protobuf_generate_cpp(ProtoSources ProtoHeaders SCAMP.proto)
add_library(messages STATIC ${ProtoSources} ${ProtoHeaders})
target_link_libraries(messages ${PROTOBUF_LIBRARY})
