#!/bin/bash
dnf -y groupinstall "Development Tools"
dnf -y install kernel-devel kernel-headers
dnf -y install cmake3 wget make unzip gcc gcc-c++
CUDA_REPO_PKG=cuda-repo-fedora${1}-10.0.130-1.x86_64.rpm
wget http://developer.download.nvidia.com/compute/cuda/repos/fedora${1}/x86_64/$CUDA_REPO_PKG
rpm -i $CUDA_REPO_PKG
dnf clean all
dnf -y install cuda
ln -s /usr/local/cuda-10.0/ /usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
cd SCAMP
mkdir build & cd build
cmake3 -D FORCE_CUDA=1 -D CMAKE_CXX_COMPILER=g++ ..
make -j4
exit 0
