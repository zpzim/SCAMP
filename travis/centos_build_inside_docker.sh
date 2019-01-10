#!/bin/bash
yum -y install epel-release
yum -y groupinstall "Development Tools"
yum -y install kernel-devel kernel-headers
yum -y install cmake3 wget autoconf automake libtool curl make unzip gcc protobuf-devel
#wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.zip
#unzip protobuf-cpp-3.6.1.zip
#pushd protobuf-3.6.1 && ./autogen.sh && ./configure --prefix=/usr CC=gcc CXX=gcc-c++ && make && make install && ldconfig && popd 
CUDA_REPO_PKG=cuda-repo-rhel${1}-10.0.130-1.x86_64.rpm
wget http://developer.download.nvidia.com/compute/cuda/repos/rhel${1}/x86_64/$CUDA_REPO_PKG
rpm -i $CUDA_REPO_PKG
yum clean all
yum -y install cuda
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -D CMAKE_CXX_COMPILER=gcc-c++ .
make -j4
exit 0
