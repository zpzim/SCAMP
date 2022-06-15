Performance
===========

SCAMP is extremely fast, especially on Tesla series GPUs. I belive this repository contains the fastest code in existance for computing the matrix profile. If you find a way to improve the speed of SCAMP, or compute matrix profiles any faster than SCAMP does, please let me know, I would be glad to point to your work and incorporate any improvements that can be made to SCAMP.

Notes on CPU performance
************************

SCAMP's CPU performance is very good. However, how performant it is depends heavily on the compiler you use. Newer compilers are better, clang v6 or greater tends to work best. Newer versions of GCC can work as well. MSVC tends to be slower. There can be up to a 10x (perhaps more) difference depending on the compiler you use. This is related to how different compilers have varying levels of support for autovectorization.

Precomputation performance
**************************

When enabling the ``--ultra_precision`` flag in the SCAMP CLI, or specifying the ``precision='ultra'`` option in pyscamp, the method for precomputing the necessary statisics for the matrix profile computation uses an O(nm) algorithm to compute the subsequence means and norms. This computation can become a bottleneck if you specify an extremely large subsequence length.

The timing results below do not use this option. All experiments were performed in double precision.

Benchmarks
**********

SCAMP has automated benchmarks running. Here is a link to recent GPU performance results:

 `GPU NVIDIA Tesla P100 (1x), Input length 1M datapoints, default parameters <https://zpzim.github.io/SCAMP/gpu-benchmarks/bench>`_ 

Note that the charts are not totally optimized for human consumption yet. But you can see a benchmark of each profile type (except PROFILE_TYPE_ALL_NEIGHBORS), the y-axis is nanoseconds.

Performance Comparisons
***********************

The included performance tests showcase SCAMP's performance up to an input size of 16M datapoints; however, as we have shown in our publications SCAMP is scalable to hundreds of millions of datapoints and even billions of datapoints with the right hardware.

.. image:: images/SCAMP_Profile_Performance_Comparison.png
  :alt: SCAMP GPU Performance

In the figure above we show the runtime in seconds for SCAMP's various profile types (self-join) on 2 P100 GPUs.

.. image:: images/KNN.png
  :alt: SCAMP KNN Performance

In the figure above we show the runtime in seconds for SCAMP's approximate KNN (``--profile_type=ALL_NEIGHBORS``) matrix profile, while varying K and the input size on 2 P100 GPUs.

You can see that SCAMP maintains good performance relative to the baseline 1NN_INDEX matrix profile up to at least K=20, which should be sufficient for almost all practioners. All measurements were made with random data with the initial threshold set to 0 correlation (close to the worst case for KNN).



Performance Comparisons with other Matrix Profile Libraries
***********************************************************

As mentioned before, SCAMP is extremely fast. This section contains experiments comparing other libraries to SCAMP in terms of performance to show quantitatively how fast SCAMP is. Note that these numbers reflect the performance of these libraries at a snapshot in time (June 2022) and implementations can change. If you want to reproduce these results, or generate new performance numbers in the future, the scripts used to generate the tables below are provided in the SCAMP repository `here <https://github.com/zpzim/SCAMP/blob/master/docs/source/scripts>`_.


.. System 1::
  WSL Ubuntu under Windows 11
  CPU: Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz
  GPU: NVIDIA GeForce RTX 3080


.. System 2:
  Linux Ubuntu 18.04
  CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
  GPU: 2x NVIDIA Tesla P100

Note: Both CPUs have SSE2/AVX/AVX2/FMA enabled.

pyscamp vs stumpy Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`stumpy <https://github.com/TDAmeritrade/stumpy>`_ is a very popular matrix profile library which has reimplemented many of the algorithms published by Eamonn's time series lab at UC Riverside.

stumpy claims to have superior performance for the matrix profile algorithms they have reimplemented. However, these performance comparisons are done in bad faith. They compare across several generations of GPU/CPU hardware instead of making a fair apples to apples comparison, they simply copy numbers published in our papers and use hardware that is multiple generations newer, and throw many more times more resources at the problem.

I contacted the stumpy maintainer years ago asking for these bad faith comparisons to be removed, but they refused. To set the record straight, here is a fair comparison of pyscamp and stumpy done on the same system.

The tables below shows that pyscamp is faster than stumpy by a factor of 20x or more on the CPU, 6x faster on GeForce GPUs, and 60x faster using Tesla Series GPUs (pyscamp is even faster here because the bottleneck on GeForce cards is fp64 compute, GeForce cards are optimized for lower precision computation). As you can see if we switch to single precision in pyscamp the GeForce performance signifigantly increases.

Both systems are using the following dependencies installed from conda-forge: ``pyscamp-gpu v4.0.0`` ``stumpy v1.11.1`` ``python v3.9.12`` ``cudatoolkit v11.6.0`` ``numba v0.55.1`` ``numpy v1.21.6`` ``scipy v1.8.1``

.. csv-table:: pyscamp vs stumpy (System 1)
   :file: data/pyscamp-vs-stumpy-cpu-and-geforce.csv
   :widths: auto
   :header-rows: 1

.. csv-table:: pyscamp vs stumpy (System 2)
   :file: data/pyscamp-vs-stumpy-cpu-and-multi-p100.csv
   :widths: auto
   :header-rows: 1

pyscamp vs MPF
^^^^^^^^^^^^^^

Matrix Profile Foundation's `matrixprofile <https://github.com/matrix-profile-foundation/matrixprofile>`_ is a python library which implements many of the matrix profile algorithms. It only supports CPU computation.

There are two algorithms in this library compared against:

* mpx: The mpx algorithm implemented in this library is very similar to what SCAMP uses and is also highly optimized, hence performance is similar here.
* SCRIMP++: I show SCRIMP++ performance here for comparison even though it is an approximate algorithm and could be made faster by changing parameters. It is a common misconception that SCRIMP++ is always faster than exact algorithms like mpx and pyscamp. There are overheads assoicated with SCRIMP++ that have high constant factor overhead (e.g. repeated FFT computation) which high-performing exact algorithms like pyscamp don't have. This can make pyscamp competetive with SCRMP++ in all but the most highly approximated scenarios.

.. csv-table:: pyscamp vs mpf (System 1)
   :file: data/pyscamp-vs-mpf-cpu.csv
   :widths: auto
   :header-rows: 1
