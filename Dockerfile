FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 AS base

RUN apt-get update && \
    apt-get upgrade -y

# SCAMP build dependancies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y zlib1g-dev cmake golang-go clang

COPY . /SCAMP

# If a build directory already exists remove it
RUN rm -rf /SCAMP/build

# Build SCAMP
RUN mkdir /SCAMP/build && cd /SCAMP/build && cmake -DSCAMP_ENABLE_BINARY_DISTRIBUTION=1 -DBUILD_CLIENT_SERVER=1 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang .. && make -j8

# We only need the CUDA runtime for the final container
FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# Copy the SCAMP binaries and tests to the final container
RUN mkdir /SCAMP
COPY --from=0 /SCAMP/build /SCAMP/build
COPY --from=0 /SCAMP/test /SCAMP/test
