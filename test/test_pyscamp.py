#This file includes tests to check the correcntess of the pyscamp module.
#It does not do extenstive testing of the SCAMP framework that is handled by other integration tests.

import pyscamp as mp
import numpy as np
from distance_matrix_fast import *
from test_common import *
import random 
import sys

failed = False
arr = []
arr2 = []

arr = np.random.random(size=(8000,))
arr2 = np.random.random(size=(8000,))

dm_self = distance_matrix(arr, None, 1024)
dm_ab = distance_matrix(arr, arr2, 1024)

dist, index = mp.selfjoin(arr, 1024, pearson=True)
dist = dist.reshape((len(dist) , 1))
index = index.reshape((len(index), 1))
vdist, vindex = reduce_1nn_index(dm_self)

if compare_vectors(vdist, dist) and compare_index(vindex, vdist, index, dist):
  print("1NN INDEX Self join pass")
else:
  failed = True
  print("1NN INDEX Self join fail")


dist, index = mp.abjoin(arr, arr2, 1024, pearson=True)
dist = dist.reshape((len(dist) , 1))
index = index.reshape((len(index), 1))
vdist, vindex = reduce_1nn_index(dm_ab)


if compare_vectors(vdist, dist) and compare_index(vindex, vdist, index, dist):
  print("1NN INDEX AB join pass")
else:
  failed = True
  print("1NN INDEX AB join fail")

dist = mp.selfjoin_sum(arr, 1024, threshold=0.90, pearson=True)
dist = dist.reshape((len(dist), 1))
vdist = reduce_sum_thresh(dm_self, 0.90)

if compare_vectors_sum(vdist, np.array(dist), 0.90):
  print("SUM Self join pass")
else:
  failed = True
  print("SUM Self join fail")


dist = mp.abjoin_sum(arr, arr2, 1024, threshold=0.90, pearson=True)
dist = dist.reshape((len(dist), 1))
vdist = reduce_sum_thresh(dm_ab, 0.90)

if compare_vectors_sum(vdist, np.array(dist), 0.90):
  print("SUM AB join pass")
else:
  failed = True
  print("SUM AB join fail")

dist = mp.selfjoin(arr, 1024, threads=1)
dist = mp.selfjoin(arr, 1024, threads=2)


if mp.gpu_supported():
  print('GPUs Supported')
  # TODO(zpzim): add a correctness check here once we have a test for that
  x = mp.selfjoin_knn(arr, 1024, 5, threshold=0.95, pearson=True)
  # TODO(zpzim): add a correctness check here once we have a test for that
  matrix = mp.abjoin_matrix(arr, arr2, 1024, threshold=0.125, mwidth=10, mheight=5, pearson=True)

if failed:
  exit(1)


