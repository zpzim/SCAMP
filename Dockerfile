FROM nvidia/cuda:10.0-devel AS base

RUN apt-get update && \
    apt-get upgrade -y

# SCAMP build dependancies
RUN apt-get install zlib1g-dev -y
RUN apt-get install cmake -y
RUN apt-get install golang-go -y
RUN apt-get install clang -y

COPY . /SCAMP

# If a build directory already exists remove it
RUN rm -rf /SCAMP/build

# Build SCAMP
RUN mkdir /SCAMP/build && cd /SCAMP/build && cmake -DBUILD_CLIENT_SERVER=1 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang .. && make -j8

# We only need the CUDA runtime for the final container
FROM nvidia/cuda:10.0-runtime

# Copy the SCAMP binaries and tests to the final container
RUN mkdir /SCAMP
COPY --from=0 /SCAMP/build /SCAMP/build
COPY --from=0 /SCAMP/test /SCAMP/test

