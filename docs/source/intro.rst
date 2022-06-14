Introduction
============

SCAMP is a CPU/GPU implementation of the SCAMP algorithm. SCAMP takes a time series as input and computes the matrix profile for a particular window size. You can read more about the matrix profile at the `Matrix Profile Homepage <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_
This is a much improved framework over `GPU-STOMP <https://github.com/zpzim/STOMPSelfJoin>`_ which has the following additional features:

 * Tiling for large inputs 
 * Computation in fp32, mixed fp32/fp64, or fp64 (double is recommended for most datasets, single precision will work for some)
 * fp32 version should get good performance on GeForce cards
 * AB joins (you can produce the matrix profile from 2 different time series)
 * Distributable on the cloud.
 * More types of matrix profiles! See `Profile Types </profiles>`.
 * Extremely Efficient Implementation
 * Extensible to adding optimized versions of custom join operations.
 * CPU Support (Only enabled for double precision; does not support KNN joins yet)
 * Handles missing data as NaN input values. The matrix profile will be computed while excluding any subsequence with a NaN value
 * Gracefully handles flat regions in data.
 * Python module: Use SCAMP in Python with pyscamp
 * conda-forge integration: Install pyscamp seamlessly with conda.
 * Extensive testing: SCAMP has thousands of input configurations tested with every PR.
 * Automatic benchmarking: Helps ensure performance doesn't slip with future updates.
 * Extremely Efficient Implementation

Motivation
==========

The matrix profile is expensive to compute. SCAMP aims to utilize specialized kernels and a tiled approach to create an extensible, scalable framework for computing the matrix profile.

Why use SCAMP?
==============
 
  * It is `faster </performance>` than any other matrix profile library. It is 10x to 100x faster than almost all other implementations out there currently.
  * It is very easy to install using conda and has very few dependencies.
  * It handles real data: very large inputs, missing values, and flat regions with little issue.
  * It can compute various other types of matrix profiles, including efficiently computing KNN matrix profiles, and matrix summaries (a.k.a. mplots). And can be extended to compute other types of profile efficiently.

When should you use SCAMP?
==========================

  * You want to go fast. :)
  * You want to compute very large matrix profiles. i.e. more than 50K-100K datapoints. The larger the dataset, the more advantage SCAMP has over other exact methods.
  * You want to compute matrix profiles using an NVIDIA GPU. With a seamless install experience.
  * You want a library that will handle real data.

When is SCAMP not the right choice?
===================================

  * SCAMP does not currently support architectures other than x86_64 (sorry Apple M1 users, you'll need to build from source). SCAMP can build on other architectures but they are not explicitly supported. Eventually support will be added but it is not currently being worked on.
  * SCAMP does not currently provide a rich API for doing things with the matrix profile once you have it. Some support for things like this is on the roadmap, but there are other libraries you can use for post processing in the meantime.
  * You want to generate matrix profiles on edge devices (sensor systems, smartwatches, raspberry pis, smartphone, etc.), these devices usually have exotic architectures (eg. 32-bits or ARM) not fully supported by SCAMP. The preference on these systems is to do some kind of approximation to reduce power usage and save on-chip resources. You might try looking into `LAMP<https://www.cs.ucr.edu/~eamonn/LAMP_Camera_Ready2.pdf>` for something like this.
