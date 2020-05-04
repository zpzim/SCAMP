pyscamp
=======

pyscamp is a python module which uses the SCAMP CUDA/C++ library to compute the matrix profile efficiently, inheriting the speed and functionality of SCAMP

Disclaimer
----------

This is a new feature and still has some kinks to work out. If you have problems building the module (or getting GPU support to work) please submit an issue on github. I don't have access to all build environments so help in addressing these issues is appreciated.



Installation
------------
A source distribution for pyscamp is available on pypi.org. The python module supports python 3 and requires that cmake is installed in order to build properly, you can install cmake using ``pip install cmake``

If you want GPU support for pyscamp, you must have CUDA installed and available to the default cmake compiler on your system. 

Please look at the :doc:`environment setup guide </environment>` for more information about how to set up the environment for SCAMP.

Once you have cmake and/or cuda installed, you can use ``pip install pyscamp`` to download and install the module.

**Installation limitations**: Currently, pyscamp has no method for you to specify a compiler/environment flags during the build phase. It will use your default cmake c++ compilers and the default cuda settings. This will soon be ammended so that you can use the configuration flags described in the environment setup guide.

Python Example
--------------

.. highlight:: python

::

  import pyscamp as mp # Uses GPU if available and CUDA was available during the build

  # Allows checking if pyscamp was built with CUDA and has GPU support
  has_gpu_support = mp.gpu_supported()

  # Self join
  profile, index = mp.selfjoin(a, sublen)
  # AB join using 4 threads
  profile, index = mp.abjoin(a, b, sublen, threads=4)
  # Sum thresh
  corr_sum = mp.abjoin_sum(a, b, sublen, threshold=0.9)

  # Approximate KNN and matrix summaries are supported with GPUs + CUDA only
  if has_gpu_support:
    knn = mp.selfjoin_knn(a,sublen, k)
    # KNN with threshold
    knn = mp.selfjoin_knn(a, sublen, k, threshold=0.85)
    # KNN Ab join with threshold, outputting pearson correlation
    knn = mp.abjoin_knn(a, b, sublen, k, threshold=0.90, pearson=True)
    # Matrix summary (100x100) with threshold, outputting pearson correlation
    matrix = mp.abjoin_matrix(a, b, sublen, mwidth=100, mheight=100, threshold=0.5, pearson=True)


