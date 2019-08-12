FROM nvidia/cuda:10.1-devel AS base

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    python \
    python-pip \
    python-virtualenv \
    zip \
    unzip \
    pxz \
    tar \
    git \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    golang \
    build-essential \
    iputils-ping 

RUN apt-get install software-properties-common -y
RUN apt-get install build-essential autoconf libtool pkg-config -y
RUN apt-get install libgflags-dev libgtest-dev -y
RUN apt-get install clang libc++-dev -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install make -y
RUN apt-get install zlib1g-dev -y
RUN apt-get install cmake -y
RUN apt-get install clang -y
RUN apt-get install module-init-tools -y

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -

RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
RUN apt-get update -y

RUN apt-get install -y cmake


ADD https://api.github.com/repos/zpzim/SCAMP/git/refs/heads/master version.json
RUN git clone --branch master https://github.com/zpzim/SCAMP.git


RUN cd /SCAMP && git submodule update --init --recursive
RUN mkdir /SCAMP/build && cd /SCAMP/build && cmake -DBUILD_CLIENT_SERVER=1 -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang .. && make -j4

