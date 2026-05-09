FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS base

# SCAMP build dependencies. Skip 'apt-get upgrade' — the base image is
# already current enough for our needs and upgrading pulls in dozens of
# unrelated packages (llvm-18 toolchain, libicu, libpython3.12, etc.) which
# multiply the chance of a transient Ubuntu mirror failure during the build.
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake zlib1g-dev clang libcub-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /SCAMP

# If a build directory already exists remove it
RUN rm -rf /SCAMP/build

# Build SCAMP
RUN mkdir /SCAMP/build && cd /SCAMP/build \
    && cmake -DSCAMP_ENABLE_BINARY_DISTRIBUTION=1 \
             -DBUILD_CLIENT_SERVER=1 \
             -DCMAKE_CXX_COMPILER=clang++ \
             -DCMAKE_C_COMPILER=clang .. \
    && make -j8

# We only need the CUDA runtime for the final container
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

RUN mkdir /SCAMP
COPY --from=base /SCAMP/build /SCAMP/build
COPY --from=base /SCAMP/test /SCAMP/test
