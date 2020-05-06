Introduction
============

SCAMP is a CPU/GPU implementation of the SCAMP algorithm. SCAMP takes a time series as input and computes the matrix profile for a particular window size. You can read more about the matrix profile at the `Matrix Profile Homepage <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_
This is a much improved framework over `GPU-STOMP <https://github.com/zpzim/STOMPSelfJoin>`_ which has the following additional features:

  * Tiling for large inputs 
  * Computation in fp32, mixed fp32/fp64, or fp64 (double is recommended for most datasets, single precision will work for some)
  * fp32 version should get good performance on GeForce cards
  * AB joins (you can produce the matrix profile from 2 different time series)
  * Distributable (we use GCP but other cloud platforms can work) with verified scalability to billions of datapoints
  * More types of matrix profiles! See Below!
  * Extremely Efficient Implementation
  * Extensible to adding optimized versions of custom join operations.
  * Can compute joins with the CPU (Only enabled for double precision and does not support all-neighbors joins or distance matrix summaries yet)
  * Handles NaN input values. The matrix profile will be computed while excluding any subsequence with a NaN value

Motivation
********** 

The matrix profile is expensive to compute. SCAMP aims to utilize specialized kernels and a tiled approach to create an extensible, scalable framework for computing the matrix profile.

