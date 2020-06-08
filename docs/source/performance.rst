Performance
===========

SCAMP is extremely fast, especially on Tesla series GPUs. I belive this repository contains the fastest code in existance for computing the matrix profile. If you find a way to improve the speed of SCAMP, or compute matrix profiles any faster than SCAMP does, please let me know, I would be glad to point to your work and incorporate any improvements that can be made to SCAMP.

Notes on CPU performance
************************

SCAMP's CPU performance is very good. However, how performant it is depends heavily on the compiler you use. Newer compilers are better, clang v6 or greater tends to work best. Newer versions of GCC can work as well. MSVC tends to be slower. There can be up to a 10x (perhaps more) difference depending on the compiler you use. This is related to how different compilers have varying levels of support for autovectorization.

Performance Comparisons
***********************

The included performance tests showcase SCAMP's performance up to an input size of 16M datapoints; however, as we have shown in our publications SCAMP is scalable to hundreds of millions of datapoints and even billions of datapoints with the right hardware.

.. image:: /images/SCAMP_Profile_Performance_Comparison.png
  :alt: SCAMP GPU Performance

In the figure above we show the runtime in seconds for SCAMP's various profile types (self-join) on 2 P100 GPUs.

.. image:: /images/KNN.png
  :alt: SCAMP KNN Performance

In the figure above we show the runtime in seconds for SCAMP's approximate KNN (--profile_type=ALL_NEIGHBORS) matrix profile, while varying K and the input size on 2 P100 GPUs.

You can see that SCAMP maintains good performance relative to the baseline 1NN_INDEX matrix profile up to at least K=20, which should be sufficient for almost all practioners. All measurements were made with random data with the initial threshold set to 0 correlation (close to the worst case for KNN).

.. image:: /images/other_methods.png
  :alt: SCAMP vs Others

The above figure illustrates SCAMP's performance versus STUMPY which is a popular matrix profile implementation. 

As can be seen above SCAMP on 2x P100 GPUs much faster than STUMPY, when STUMPY is running on 16x V100 GPUs, which are about ~2x more powerful than P100s individually. This is several orders of magnitude of difference in processing power.

